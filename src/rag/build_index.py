"""
Build the RAG document corpus from kb/dishes.json.

Outputs (relative to src/rag/):
  corpus.json       — list of text documents, one per dish
  seq_to_name.json  — {seq_id: dish_name} mapping (seq_id is 1-indexed)

Run from src/rag/:
    python build_index.py

Or import and call build() programmatically.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
KB_PATH = HERE / ".." / "kb" / "dishes.json"
CORPUS_PATH = HERE / "corpus.json"
SEQ_TO_NAME_PATH = HERE / "seq_to_name.json"


def _format_licenses(restaurant: dict) -> str:
    parts = []
    ltk = restaurant.get("ltk")
    if ltk:
        parts.append(f"LTK {ltk}")
    for name, level in (restaurant.get("licenses") or {}).items():
        if level:
            parts.append(f"{name} {level}")
    return ", ".join(parts) if parts else "nessuna"


def _format_dish(seq_id: int, dish: dict, restaurant: dict) -> str:
    ing = ", ".join(dish.get("ingredients") or []) or "—"
    tech = ", ".join(dish.get("techniques") or []) or "—"
    planet = restaurant.get("planet") or "sconosciuto"
    licenses = _format_licenses(restaurant)

    # Il seguente formato [N] testo del piatto con formattazione specifica serve poi in parse response per estrarre le informazioni dal documento in cui si trova l'informazione ricercata nella query.
    return (
        f"[{seq_id}] Piatto: {dish['name']}\n"
        f"Ristorante: {restaurant['restaurant']} | Pianeta: {planet}\n"
        f"Chef: {restaurant.get('chef', '—')} | Licenze: {licenses}\n"
        f"Ingredienti: {ing}\n"
        f"Tecniche: {tech}"
    )


# Avendo costruito la kb (dalla soluzione no rag), la si sfrutta per avere un corpus unico di testo che sarà utilizzato per il retriever.
def build() -> tuple[list[str], dict[int, str]]:
    """Build corpus and seq→name mapping from dishes.json.

    Returns:
        corpus: list of document strings (index 0 = seq_id 1)
        seq_to_name: {seq_id: dish_name}
    """
    with open(KB_PATH, encoding="utf-8") as f:
        data = json.load(f)

    corpus: list[str] = []
    seq_to_name: dict[int, str] = {}
    seq_id = 1

    for restaurant in data["restaurants"]:
        for dish in restaurant.get("dishes") or []:
            doc = _format_dish(seq_id, dish, restaurant)
            corpus.append(doc)
            seq_to_name[seq_id] = dish["name"]
            seq_id += 1

    CORPUS_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    SEQ_TO_NAME_PATH.write_text(
        json.dumps(seq_to_name, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Built index: {len(corpus)} documenti")
    print(f"  corpus     -> {CORPUS_PATH}")
    print(f"  seq_to_name -> {SEQ_TO_NAME_PATH}")
    return corpus, seq_to_name


if __name__ == "__main__":
    build()
