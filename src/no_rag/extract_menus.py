import json
import os
import re
import sys
import time
from pathlib import Path
import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import API_KEY_ENV_NAME, GROQ_MODEL_LLAMA_INSTANT

from groq import Groq

MENU_DIR = Path("../../Dataset/knowledge_base_md/menu")
KB_DIR = Path("../kb")
MENUS_CACHE_DIR = KB_DIR / "menus"

DISH_BATCH_SIZE = 3

# Serie di prompt definiti da Claude che ha letto tutti i menu. Il prompt potrebbe dipendere molto da come questi sono strutturati così come non potrebbe.
# TODO: Come valutazione a posteriori sulla robustezza del sistema si potrebbe fare un'analisi progressiva (monitorando magari la jaccard similarity) di come il prompt cambia man mano che si aggiungono menù e di come, all'aumentare del numero di file, il prompt precedente sia più o meno robusto al cambiamento.
# La difficoltà principale del parsing dei menù è quella di dover ricorrere ad un LLM vista la loro struttura non prevedibile e varia. Questo significa dover sapere il limite di token che l'LLM di turno ha ed eventualmente proporre delle strategie di chunking. In questo caso, essendo l'LLM utilizzato molto semplice e leggero (e quindi utilizzabile più a lungo per API gratuitamente) si è deciso di dividere il documento in due macro chunk, introduzione con metadati del ristorante + piatti, e questi ultimi divisi in chunk da 3 piatti l'uno
SYSTEM_PROMPT = (
    "You are a precise data extractor for restaurant menus. "
    "Return ONLY valid JSON with no markdown, no explanations."
)

METADATA_PROMPT = """Extract the restaurant metadata from this menu header section.

Rules:
- License levels: convert Roman numerals to integers (I=1, II=2, III=3, IV=4, V=5, IX=9, X=10, XI=11, XVI=16).
  Numbers stay as-is.
- Defaults when not mentioned: Psionica=0, Gravitazionale=0, Antimateria=0;
  Temporale/Magnetica/Quantistica/Luce/LTK = null.
- LTK: extract the integer from "LTK", "livello tecnologico", or "EDUCATION di livello tecnologico"; null if absent.
- Planet: look for "sul pianeta X", "su X", "del pianeta X"; null if absent.
- Ordini: list professional orders the restaurant/chef belongs to
  (Andromeda, Naturalisti, Armonisti); empty list if none.
- "Education Level" or "EDUCATION" in some menus equals the license level.
- Restaurant name: the actual restaurant name, without surrounding quotes.

Return ONLY this JSON (no extra keys):
{
  "restaurant": "exact restaurant name",
  "chef": "chef full name",
  "planet": "planet name or null",
  "licenses": {
    "Psionica": 0,
    "Temporale": null,
    "Gravitazionale": 0,
    "Antimateria": 0,
    "Magnetica": null,
    "Quantistica": null,
    "Luce": null
  },
  "ltk": null,
  "ordini": []
}

MENU HEADER:
{metadata_text}
"""

DISHES_PROMPT = """Extract all dish data from these restaurant menu entries.

Rules:
1. Ingredient and technique names: copy EXACTLY as written — do NOT translate or paraphrase.
2. Doppia tecnica: if a technique is used TWICE in one dish (e.g. "doppia Cottura a Vapore ..."),
   include it TWICE in the list.
   Example: "doppia Cottura a Vapore Ecodinamico Bilanciato"
   → techniques: ["Cottura a Vapore Ecodinamico Bilanciato", "Cottura a Vapore Ecodinamico Bilanciato"]
3. Ordine: strip emoji from the dish name and record in "ordine":
   🪐 → "Andromeda", 🌱 → "Naturalisti", 🌈 → "Armonisti", no emoji → null
4. Collect ALL ingredients/techniques from BOTH the prose AND any Ingredienti/Tecniche sections.
5. When multiple items appear on one line without separators, split them into individual items.
6. Ignore any non-dish section headers (e.g. "Ordini Professionali") — only extract actual dishes.

Return ONLY this JSON:
{
  "dishes": [
    {
      "name": "dish name without emoji",
      "ordine": null,
      "ingredients": ["exact ingredient name"],
      "techniques": ["exact technique name"]
    }
  ]
}

MENU ENTRIES:
{dishes_text}
"""

# Headers che non sono piatti
_NON_DISH_HEADERS = {"menu", "ingredienti", "tecniche"}

# I livelli  livelli di licenza dei ristoranti/chef che sono in numeri romani
def roman_to_int(s: str) -> int | None:
    table = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
        "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
        "XI": 11, "XII": 12, "XIV": 14, "XVI": 16,
    }
    s = s.strip().upper()
    if s in table:
        return table[s]
    try:
        return int(s)
    except ValueError:
        return None


def _header_text(line: str) -> str:
    return re.sub(r"^#+\s*[\*_]*", "", line.strip()).strip().strip("*_").strip()


def split_menu_into_blocks(text: str) -> tuple[str, list[str]]:
    """Split menu in (metadata_block, [dish_block_1, ...]).

    metadata_block: tutto ciò che è presente sopra il separatore ## Menu. Se questo separatore non c'è, come nel caso Datapizza, si considera che i metadati sono quelli prima del secondo header. Per come sono strutturati i menu a disposizione.

    dish_blocks: una stringa per piatto, ognuna che inizia con l'header che è il nome del piatto. Le sottosezioni sono gli ingredienti e le tecniche e sono tenute dentro il blocco del piatto.
    """
    lines = text.splitlines()

    # Ricerca del separatore ## Menu
    menu_line_idx: int | None = None
    for i, line in enumerate(lines):
        if line.strip().startswith("##"):
            ht = _header_text(line).lower()
            if ht == "menu":
                menu_line_idx = i
                break

    if menu_line_idx is not None:
        start_idx = menu_line_idx + 1
        metadata_block = "\n".join(lines[:start_idx])
    else:
        # Se non c'è (caso Datapizza), considero tutto ciò che c'è prima
        count = 0
        start_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith("##"):
                count += 1
                if count == 2:
                    start_idx = i
                    break
        metadata_block = "\n".join(lines[:start_idx])

    # Costruzione dei dish blocks
    dish_blocks: list[str] = []
    current: list[str] = []

    for line in lines[start_idx:]:
        if line.strip().startswith("##"):
            ht = _header_text(line).lower()
            if ht not in _NON_DISH_HEADERS and ht:
                if current:
                    dish_blocks.append("\n".join(current))
                current = [line]
            else:
                if current:
                    current.append(line)
        else:
            if current:
                current.append(line)

    if current:
        dish_blocks.append("\n".join(current))

    return metadata_block, dish_blocks


def call_groq(client: Groq, prompt: str) -> dict:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_LLAMA_INSTANT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                print(f"    Retry {attempt + 1}/3 after: {e}")
                time.sleep(5 * (attempt + 1))
            else:
                raise


def extract_metadata(client: Groq, metadata_block: str) -> dict:
    prompt = METADATA_PROMPT.replace("{metadata_text}", metadata_block)
    result = call_groq(client, prompt)
    time.sleep(3.0)
    return result


def extract_dishes_batch(client: Groq, dish_blocks: list[str]) -> list[dict]:
    dishes_text = "\n\n---\n\n".join(dish_blocks)
    prompt = DISHES_PROMPT.replace("{dishes_text}", dishes_text)
    result = call_groq(client, prompt)
    dishes = result.get("dishes", [])
    time.sleep(3.0)
    if len(dishes) < len(dish_blocks):
        print(f"    Note: {len(dish_blocks)} blocchi → {len(dishes)} piatti estratti (blocchi che non sono piatti sono ignorati)")
    return dishes


def normalize_result(data: dict) -> dict:
    """Normalize license levels, validate ordine, ensure list fields."""
    licenses = data.get("licenses", {})
    # Licenze di default
    defaults: dict[str, int | None] = {
        "Psionica": 0,
        "Temporale": None,
        "Gravitazionale": 0,
        "Antimateria": 0,
        "Magnetica": None,
        "Quantistica": None,
        "Luce": None,
    }
    normalized_licenses: dict[str, int | None] = {}
    # Le licenze estratte dovrebbero essere tra quelle di default.
    # I valori sono quelli che cambiano, in base al livello richiesto
    for key, default in defaults.items():
        val = licenses.get(key, default)
        if isinstance(val, str):
            converted = roman_to_int(val)
            val = converted if converted is not None else default
        normalized_licenses[key] = val

    # LTK potrebbe essere nel dizionario delle licenze
    if "LTK" in licenses:
        val = licenses["LTK"]
        if isinstance(val, str):
            val = roman_to_int(val)
        data["ltk"] = val

    data["licenses"] = normalized_licenses

    ltk = data.get("ltk")
    if isinstance(ltk, str):
        data["ltk"] = roman_to_int(ltk)

    if not isinstance(data.get("dishes"), list):
        data["dishes"] = []

    # Costruzione della lista dei piatti per completare il json con i dati del ristorante
    for dish in data["dishes"]:
        if not isinstance(dish.get("ingredients"), list):
            dish["ingredients"] = []
        if not isinstance(dish.get("techniques"), list):
            dish["techniques"] = []

        dish["ingredients"] = [i.strip() for i in dish["ingredients"] if str(i).strip()]
        dish["techniques"] = [t.strip() for t in dish["techniques"] if str(t).strip()]

        if dish.get("ordine") not in ("Andromeda", "Naturalisti", "Armonisti"):
            dish["ordine"] = None

    if not isinstance(data.get("ordini"), list):
        data["ordini"] = []

    return data


def process_menu(client: Groq, menu_file: Path) -> dict:
    content = menu_file.read_text(encoding="utf-8")
    metadata_block, dish_blocks = split_menu_into_blocks(content)

    n_batches = -(-len(dish_blocks) // DISH_BATCH_SIZE)  # ceil division
    print(f"    Blocchi: 1 metadata + {len(dish_blocks)} piatti ({n_batches} batches)")

    metadata = extract_metadata(client, metadata_block)

    all_dishes: list[dict] = []
    for i in range(0, len(dish_blocks), DISH_BATCH_SIZE):
        batch = dish_blocks[i : i + DISH_BATCH_SIZE]
        print(f"    Batch {i // DISH_BATCH_SIZE + 1}/{n_batches}")
        dishes = extract_dishes_batch(client, batch)
        all_dishes.extend(dishes)

    # Questo è costruito sulla base delle domande. La struttura è stata proposta da Claude sulla base delle domande presentate. Si presuppone che, data la quantità di file e domande presenti, l'eventuale introduzione di altri file di ristoranti o di altre domande non sia necessario ritoccare la seguente struttura in quanto ciò sarebbe un cambiamento non insignificante nei requisiti.
    # La seguente struttura è un po' come una tabella di un database. Visto quanto i filtri poi si baseranno su questa struttura direi che come similarità si è maggiormente di fronte ad un db SQL piuttosto che NO-SQL, visto quanto i dati devono essere strutturati.
    result: dict = {
        "restaurant": metadata.get("restaurant", ""),
        "chef": metadata.get("chef", ""),
        "planet": metadata.get("planet"),
        "licenses": metadata.get("licenses", {}),
        "ltk": metadata.get("ltk"),
        "ordini": metadata.get("ordini", []),
        "dishes": all_dishes,
    }

    return normalize_result(result)


def process_all_menus() -> None:
    api_key = os.environ.get(API_KEY_ENV_NAME)
    if not api_key:
        raise ValueError(
            "Variabile d'ambiente GROQ_API_KEY non impostata.\n"
            "Get a free key at https://console.groq.com"
        )

    client = Groq(api_key=api_key)
    KB_DIR.mkdir(exist_ok=True)
    MENUS_CACHE_DIR.mkdir(exist_ok=True)

    menu_files = sorted(MENU_DIR.glob("*.md"))
    print(f"Trovati {len(menu_files)} menu files\n")

    all_restaurants: list[dict] = []

    for i, menu_file in enumerate(menu_files, 1):
        cache_file = MENUS_CACHE_DIR / f"{menu_file.stem}.json"

        # utilizzo un meccanismo di caching per evitare che, in caso di errore di API o interruzione del parsing per qualsiasi motivo, non si renda necessario ricominciare da capo.
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            if "error" in data:
                print(f"[{i:02d}/{len(menu_files)}] Errore di cache, re-processing: {menu_file.name}")
            else:
                print(f"[{i:02d}/{len(menu_files)}] Utilizzo della cache: {menu_file.name}")
                all_restaurants.append(data)
                continue

        print(f"[{i:02d}/{len(menu_files)}] Estrazione di: {menu_file.name}")

        try:
            data = process_menu(client, menu_file)
            n_dishes = len(data.get("dishes", []))
            print(
                f"    → {data.get('restaurant', '?')} | "
                f"planet={data.get('planet')} | "
                f"dishes={n_dishes}"
            )

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            all_restaurants.append(data)

        except Exception as e:
            print(f"    ERROR: {e}")
            cache_file.write_text(
                json.dumps({"error": str(e), "file": menu_file.name}),
                encoding="utf-8",
            )

    output = {"restaurants": all_restaurants}
    out_path = KB_DIR / "dishes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_dishes = sum(len(r.get("dishes", [])) for r in all_restaurants)
    print(f"\nDone! {len(all_restaurants)} ristoranti, {total_dishes} piatti totali")
    print(f"Salvato in {out_path}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    process_all_menus()
