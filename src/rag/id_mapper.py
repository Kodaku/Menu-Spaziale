import difflib
import json
import re

# Elimino spazi bianchi e metto tutto in lowercase
def _norm_exact(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# Rimuovo anche eventuale punteggiatura
def _norm_fuzzy(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

# Sono considerati 3 tipi di mapping diversi per evitare di andare incontro a situazioni in cui l'LLM estrae il nome di un piatto in modo diverso a quello presente nei ground truth:
# 1. Matching esatto.
# 2. Matching tutto lower case
# 3. Matching fuzzy, per diffstring
class DishMapper:
    def __init__(self, mapping_path: str):
        with open(mapping_path, encoding="utf-8") as f:
            self._mapping: dict[str, int] = json.load(f)

        # Tabelle di lookup per mappare correttamente i nomi dei piatti
        self._exact_lower: dict[str, int] = {
            _norm_exact(k): v for k, v in self._mapping.items()
        }
        self._names: list[str] = list(self._mapping.keys())
        self._fuzzy_keys: list[str] = [_norm_fuzzy(k) for k in self._names]

    def get_id(self, name: str) -> int | None:
        # 1. Nome esatto, presente anche nella ground truth
        if name in self._mapping:
            return self._mapping[name]
        # 2. Eventualmente provo un match case insensitive
        lower = _norm_exact(name)
        if lower in self._exact_lower:
            return self._exact_lower[lower]
        # 3. Provo eventualmente a rimuovere la punteggiatura e a fare un match per similarità, con string difference
        fuzz = _norm_fuzzy(name)
        close = difflib.get_close_matches(fuzz, self._fuzzy_keys, n=1, cutoff=0.85)
        if close:
            idx = self._fuzzy_keys.index(close[0])
            return self._mapping[self._names[idx]]
        return None

    def map_names(self, names: list[str]) -> list[int]:
        seen: set[int] = set()
        ids: list[int] = []
        for name in names:
            dish_id = self.get_id(name)
            if dish_id is not None and dish_id not in seen:
                seen.add(dish_id)
                ids.append(dish_id)
        return sorted(ids)
