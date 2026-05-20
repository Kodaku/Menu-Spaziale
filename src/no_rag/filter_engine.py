"""
Deterministic filter engine: applies a structured filter dict against dishes.json.
"""

import csv
from collections import Counter
import json
from pathlib import Path

KB_PATH = Path("../kb/dishes.json")

_HERE = Path(__file__).resolve().parent
_CAT_PATH = _HERE.parent / "kb" / "tecnica_to_categoria.json"
_DISTANZE_PATH = _HERE.parent.parent / "Dataset" / "knowledge_base" / "misc" / "Distanze.csv"

# Short license codes → canonical key in restaurant.licenses
_SHORT_TO_CANONICAL = {
    "p": "Psionica",
    "q": "Quantistica",
    "g": "Gravitazionale",
    "mx": "Magnetica",
    "t": "Temporale",
    "e+": "Antimateria",
}
_LONG_TO_CANONICAL = {
    "psionica": "Psionica",
    "temporale": "Temporale",
    "gravitazionale": "Gravitazionale",
    "antimateria": "Antimateria",
    "luce": "Luce",
    "magnetica": "Magnetica",
    "quantistica": "Quantistica",
    "ltk": "ltk",
}


def _norm(s: str) -> str:
    return s.strip().lower()


def _contains_term(entity_list: list[str], term: str) -> bool:
    """Return True if term matches any entity exactly or as a word-suffix.

    Handles cases like entity="tanto ravioli al vaporeon", term="ravioli al vaporeon".
    """
    for entity in entity_list:
        if entity == term or entity.endswith(" " + term):
            return True
    return False


def _resolve_license(name: str) -> str:
    lower = _norm(name)
    if lower in _SHORT_TO_CANONICAL:
        return _SHORT_TO_CANONICAL[lower]
    return _LONG_TO_CANONICAL.get(lower, name)


def _load_categories() -> dict[str, set[str]]:
    """Return {normalized_category: {normalized_technique, ...}}."""
    cat_to_techs: dict[str, set[str]] = {}
    if not _CAT_PATH.exists():
        return cat_to_techs
    with open(_CAT_PATH, encoding="utf-8") as f:
        tech_to_cat: dict[str, str] = json.load(f)
    for tech, cat in tech_to_cat.items():
        cat_to_techs.setdefault(_norm(cat), set()).add(_norm(tech))
    return cat_to_techs


def _load_distances() -> dict[str, dict[str, int]]:
    """Return {normalized_planet: {normalized_planet: distance_ly}}."""
    distances: dict[str, dict[str, int]] = {}
    if not _DISTANZE_PATH.exists():
        return distances
    with open(_DISTANZE_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items = list(row.items())
            source = _norm(items[0][1])  # value of first column (planet name)
            distances[source] = {}
            for dest_col, dist_str in items[1:]:
                distances[source][_norm(dest_col)] = int(dist_str)
    return distances


_TECH_CATEGORIES: dict[str, set[str]] = _load_categories()
_DISTANCES: dict[str, dict[str, int]] = _load_distances()


def _restaurant_passes(restaurant: dict, filters: dict) -> bool:
    # X_f sta per X del filtro
    planet_f = filters.get("planet")
    # Cerco l'eventuale pianeta, se richiesto dal filtro
    if planet_f:
        planet = _norm(restaurant.get("planet") or "")
        # Se il pianeta non matcha allora il ristorante che cerco non è quello
        if _norm(planet_f) not in planet:
            return False

    rest_f = filters.get("restaurant_name")
    if rest_f:
        r_name = _norm(restaurant.get("restaurant") or "")
        rf_norm = _norm(rest_f)
        # Se il ristorante non matcha allora non è quello che cerco
        if rf_norm not in r_name and r_name not in rf_norm:
            return False
    # Filtri delle licenze
    for lic_name, min_level in (filters.get("min_licenses") or {}).items():
        key = _resolve_license(lic_name)
        if key == "ltk":
            level = restaurant.get("ltk")
        else:
            level = (restaurant.get("licenses") or {}).get(key)
        if level is None or level < min_level:
            return False

    # Filtri delle distanze
    max_dist_f = filters.get("max_distance_from")
    if max_dist_f and _DISTANCES:
        origin = _norm(max_dist_f.get("planet") or "")
        max_ly = max_dist_f.get("max_ly", 0)
        rest_planet = _norm(restaurant.get("planet") or "")
        if origin and rest_planet:
            origin_dists = _DISTANCES.get(origin, {})
            dist = origin_dists.get(rest_planet)
            # Cerco il ristorante che matcha con la distanza voluta dal filtro
            if dist is None or dist > max_ly:
                return False
    # Se arrivo qui il filtro è passato
    return True


def _dish_matches(dish: dict, filters: dict) -> bool:
    dish_ing = [_norm(i) for i in dish.get("ingredients") or []]
    dish_tech = [_norm(t) for t in dish.get("techniques") or []]

    # Filtraggio per ingredienti, inclusi. Quindi per filtri che recitano: "Con Chocobo wings"
    inc_ing = [_norm(i) for i in filters.get("include_ingredients") or []]
    if inc_ing:
        # Il piatto deve contenere gli ingrediente del filtro
        if filters.get("require_all_ingredients", True):
            if not all(_contains_term(dish_ing, i) for i in inc_ing):
                return False
        else:
            if not any(_contains_term(dish_ing, i) for i in inc_ing):
                return False

    # Filtraggio per ingredienti esclusi: per domande che recitano: "senza chcocobo wings"
    excl_ing = [_norm(i) for i in filters.get("exclude_ingredients") or []]
    if any(_contains_term(dish_ing, i) for i in excl_ing):
        return False

    # Se si richiede una doppia tecnica questa è contata due volte nel filtro con il suo nome originale. Comunque si parla di inclusione di tecniche
    inc_tech = [_norm(t) for t in filters.get("include_techniques") or []]
    if inc_tech:
        if filters.get("require_all_techniques", True):
            required = Counter(inc_tech)
            for req_term, req_count in required.items():
                actual = sum(1 for dt in dish_tech if _contains_term([dt], req_term))
                if actual < req_count:
                    return False
        else:
            if not any(_contains_term(dish_tech, t) for t in set(inc_tech)):
                return False

    # Escludi tecniche
    excl_tech = [_norm(t) for t in filters.get("exclude_techniques") or []]
    if any(_contains_term(dish_tech, t) for t in excl_tech):
        return False

    # Per filtri: "Almeno un..."
    min_count = filters.get("min_count_from_list")
    candidates = [_norm(c) for c in filters.get("candidate_list") or []]
    if min_count is not None and candidates:
        matched = sum(
            1 for c in candidates
            if _contains_term(dish_ing, c) or _contains_term(dish_tech, c)
        )
        # Match di "Almeno".
        if matched < min_count:
            return False

    # Filtro per categorie di tecniche
    req_cats = [_norm(c) for c in filters.get("technique_categories") or []]
    if req_cats and _TECH_CATEGORIES:
        for cat in req_cats:
            techs_in_cat = _TECH_CATEGORIES.get(cat, set())
            if not any(t in techs_in_cat for t in dish_tech):
                return False
    # Se arrivo qui in fondo il filtro è passato
    return True


def filter_dishes(db: dict, filters: dict) -> list[str]:
    results: list[str] = []
    for restaurant in db.get("restaurants", []):
        # Cerco il ristorante richiesto dal filtro, se presente. Se non c'è skippo
        if not _restaurant_passes(restaurant, filters):
            continue
        for dish in restaurant.get("dishes", []):
            if _dish_matches(dish, filters):
                results.append(dish["name"])
    # Risultato dal filtraggio. Sarebbe come lanciare una query sul db
    return results


def load_db(kb_path: Path = KB_PATH) -> dict:
    with open(kb_path, encoding="utf-8") as f:
        return json.load(f)
