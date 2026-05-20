"""
Parse natural-language questions into structured filter dicts using an LLM callable.
"""

import json
import re
from pathlib import Path
from typing import Callable

KB_PATH = Path(__file__).resolve().parent.parent / "kb" / "dishes.json"
TECNICA_TO_CAT_PATH = Path(__file__).resolve().parent.parent / "kb" / "tecnica_to_categoria.json"

# Filtri disponibili, valutati in base alle domande presenti.
# TODO: La struttura non è di per sé robusta, è molto dipendente dalla tipologia di domande. Aggiungendo nuove domande potrebbe essere necessario definire nuovi filtri. Il drawback è lo stesso di quanto si definiscono i filtri di un sito web.
_FILTER_DEFAULTS: dict = {
    "include_ingredients": [],
    "exclude_ingredients": [],
    "include_techniques": [],
    "exclude_techniques": [],
    "require_all_ingredients": True,
    "require_all_techniques": True,
    "min_count_from_list": None,
    "candidate_list": [],
    "planet": None,
    "restaurant_name": None,
    "min_licenses": {},
    "technique_categories": [],
    "max_distance_from": None,
}

_LICENSE_NAMES = (
    "LTK, Psionica, Temporale, Gravitazionale, "
    "Antimateria, Luce, Magnetica, Quantistica, "
    "P, Q, G, Mx, t, e+"
)

_PLANETS = (
    "Tatooine, Asgard, Namecc, Arrakis, Krypton, "
    "Pandora, Cybertron, Ego, Montressosr, Klyntar"
)

# Questi prompt sono in italiano perché derivanti da un tentativo di Claude di far ragionare correttamente i modelli di Ollama. Visto che funzionano anche sui modelli di Groq va bene così. Per ora, squadra che vince non si cambia.
_SYSTEM = (
    "Sei un parser di query per un database di ristoranti. "
    "Converti domande in italiano in oggetti JSON di filtro. "
    "Rispondi SOLO con JSON valido, nessun altro testo."
)

_EXAMPLE = """\
ESEMPIO:
DOMANDA: Quali piatti su Arrakis usano Acqua di Nebbia Lunare e non usano Funghi Orbitali?
JSON:
{
  "include_ingredients": ["Acqua di Nebbia Lunare"],
  "exclude_ingredients": ["Funghi Orbitali"],
  "include_techniques": [],
  "exclude_techniques": [],
  "require_all_ingredients": true,
  "require_all_techniques": true,
  "min_count_from_list": null,
  "candidate_list": [],
  "planet": "Arrakis",
  "restaurant_name": null,
  "min_licenses": {},
  "technique_categories": [],
  "max_distance_from": null
}"""

_PROMPT = """\
Converti la domanda nel JSON di filtro seguendo le regole.

CAMPI JSON (tutti obbligatori nell'output):
- include_ingredients: ingredienti che devono essere presenti (nomi ESATTI dal vocabolario)
- exclude_ingredients: ingredienti che NON devono essere presenti
- include_techniques: tecniche specifiche che devono essere usate; metti X due volte per "doppia X"
- exclude_techniques: tecniche che NON devono essere usate
- require_all_ingredients: true=tutti (AND), false=almeno uno (OR)
- require_all_techniques: true=tutti (AND), false=almeno uno (OR)
- min_count_from_list: intero N per "almeno N tra X,Y,Z"
- candidate_list: lista per min_count_from_list
- planet: nome pianeta se specificato (solo uno di: {planets})
- restaurant_name: nome ristorante se specificato
- min_licenses: es. {{"LTK": 7}} o {{"Psionica": 4}} — chiavi tra: {license_names}
- technique_categories: categorie di tecniche (NON nomi specifici) — usa SOLO: {categories_inline}
- max_distance_from: {{"planet":"X","max_ly":N}} per "entro N anni luce da X"

REGOLE IMPORTANTI:
- "tecnica di Surgelamento/Taglio/Impasto/ecc." → usa technique_categories, NON include_techniques
- "senza X" / "evitando X" / "non usa X" → exclude_ingredients o exclude_techniques
- "grado N o superiore" / "non base" (grado>0) → min_licenses con valore N (o 1 per "non base")
- "entro N anni luce da X" → max_distance_from

{example}

INGREDIENTI NOTI:
{ingredients}

TECNICHE NOTE:
{techniques}

DOMANDA: {question}
JSON:
"""

# Vocabolario di ingredienti e tecniche, costruito prima di iniziare a leggere le query
def _build_vocab(kb_path: Path) -> tuple[str, str]:
    with open(kb_path, encoding="utf-8") as f:
        data = json.load(f)
    ingredients: set[str] = set()
    techniques: set[str] = set()
    for r in data["restaurants"]:
        for d in r["dishes"]:
            for i in d.get("ingredients") or []:
                s = i.strip()
                if s:
                    ingredients.add(s)
            for t in d.get("techniques") or []:
                s = t.strip()
                if s:
                    techniques.add(s)
    ing_str = "\n".join(f"  - {i}" for i in sorted(ingredients))
    tech_str = "\n".join(f"  - {t}" for t in sorted(techniques))
    return ing_str, tech_str

# Lettura del file tecnica_to_categoria.json
def _build_categories_str(cat_path: Path) -> str:
    if not cat_path.exists():
        return "  (file not found)"
    with open(cat_path, encoding="utf-8") as f:
        tech_to_cat: dict[str, str] = json.load(f)
    cats: set[str] = set(tech_to_cat.values())
    return "\n".join(f"  - {c}" for c in sorted(cats))


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON in LLM response: {text[:200]!r}")


class QueryParser:
    def __init__(
        self,
        call: Callable,
        kb_path: Path = KB_PATH,
        cat_path: Path = TECNICA_TO_CAT_PATH,
    ):
        self._call = call
        ing_str, tech_str = _build_vocab(kb_path)
        self._ing = ing_str
        self._tech = tech_str
        self._cats = _build_categories_str(cat_path)
        # Lista di categorie separate da virgola
        self._cats_inline = ", ".join(
            line.strip().lstrip("- ") for line in self._cats.splitlines() if line.strip()
        )
        # Normalizzazione dei nomi di categoria mettendoli lowercase
        self._cat_names: set[str] = {
            line.strip().lstrip("- ").lower()
            for line in self._cats.splitlines()
            if line.strip()
        }

    def parse(self, question: str) -> dict:
        prompt = _PROMPT.format(
            license_names=_LICENSE_NAMES,
            planets=_PLANETS,
            categories_inline=self._cats_inline,
            example=_EXAMPLE,
            ingredients=self._ing,
            techniques=self._tech,
            question=question,
        )
        raw_text = self._call(prompt, _SYSTEM, 512)
        # Tentativo di parsing del json. L'output è davvero un json?
        raw = _parse_json(raw_text)
        result = dict(_FILTER_DEFAULTS)
        result.update(raw)

        # Mapping dalle tecniche incluse alle sue categorie
        # Questo per coprire casistiche come: "piatti che includono tecniche di surgelamento". "Surgelamento" non è una tecnica ma una categoria ma l'LLM potrebbe trattarlo come una tecnica vera e propria. Con questo codice tutto torna al suo posto.
        if self._cat_names:
            clean, promoted = [], []
            for t in result.get("include_techniques") or []:
                (promoted if t.lower() in self._cat_names else clean).append(t)
            if promoted:
                result["include_techniques"] = clean
                existing = result.get("technique_categories") or []
                result["technique_categories"] = list(dict.fromkeys(existing + promoted))

        return result
