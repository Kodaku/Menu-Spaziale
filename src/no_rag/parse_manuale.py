import json
import re
from pathlib import Path

MANUALE_PATH = Path("../../Dataset/knowledge_base_md/misc/Manuale di Cucina.md")
KB_DIR = Path("../kb")

# Header delle categorie
CATEGORY_HEADERS = {
    "Marinatura",
    "Affumicatura",
    "Fermentazione",
    "Tecniche di Impasto",
    "Surgelamento",
    "Bollitura",
    "Grigliare",
    "Forno",
    "Vapore",
    "Sottovuoto",
    "Saltare in Padella",
    "Decostruzione",
    "Sferificazione",
    "Tecniche di Taglio",
}

# Mappo i nomi delle categorie in nomi che poi sono utilizzati nelle query
HEADER_TO_CANONICAL = {
    "Marinatura": "Marinatura",
    "Affumicatura": "Affumicatura",
    "Fermentazione": "Fermentazione",
    "Tecniche di Impasto": "Impasto",
    "Surgelamento": "Surgelamento",
    "Bollitura": "Bollitura",
    "Grigliare": "Grigliatura",
    "Forno": "Cottura al Forno",
    "Vapore": "Cottura al Vapore",
    "Sottovuoto": "Cottura Sottovuoto",
    "Saltare in Padella": "Cottura al Salto",
    "Decostruzione": "Decostruzione",
    "Sferificazione": "Sferificazione",
    "Tecniche di Taglio": "Taglio",
}

# Skippo tutti gli headers che non sono categorie
SKIP_PATTERNS = re.compile(
    r"^(Ca itolo|Introduzione|Psionica|Temporale|Antimateria|Magnetica|"
    r"Quantistica|Luce|Livello di Sviluppo|Gravitazionale|Ordine|"
    r"🪐|🌱|🌈|APPENDICE|CALCOLO)",
    re.IGNORECASE,
)


def parse_manuale(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    mapping: dict[str, str] = {}
    current_category: str | None = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Skippo tutto ciò che non è header
        # TODO: forse una strategia debole se il manuale in md contiene
        # errori di conversione da pdf
        if not line.startswith("##"):
            continue

        # Testo dell'header
        header = re.sub(r"^#+\s*\*?\*?", "", line).strip().strip("*").strip()

        if not header:
            continue

        # Salto tutti i pattern che non sono contemplati
        if SKIP_PATTERNS.match(header):
            continue

        # Header accettato -> ne faccio il mapping
        if header in CATEGORY_HEADERS:
            current_category = HEADER_TO_CANONICAL[header]
            continue

        if current_category is None:
            continue

        # Gli header che sono tecniche sono succeduti (?) dalla frase "come funziona". Se questa c'è allora l'header rappresenta una tecnica
        for j in range(i + 1, min(i + 5, len(lines))):
            next_line = lines[j].strip()
            if next_line:
                if next_line.lower().startswith("come funziona"):
                    mapping[header] = current_category
                break

    return mapping


def build_tecnica_to_categoria() -> None:
    KB_DIR.mkdir(exist_ok=True)

    mapping = parse_manuale(MANUALE_PATH)

    out_path = KB_DIR / "tecnica_to_categoria.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Salvati {len(mapping)} mappings tecnica→categoria in{out_path}")

    # Verifica
    by_cat: dict[str, list[str]] = {}
    for tech, cat in mapping.items():
        by_cat.setdefault(cat, []).append(tech)
    for cat, techs in sorted(by_cat.items()):
        print(f"  {cat}: {len(techs)} tecniche")


if __name__ == "__main__":
    build_tecnica_to_categoria()
