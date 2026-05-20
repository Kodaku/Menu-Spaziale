#!/usr/bin/env python3
"""Parse blog HTML files to extract ingredient percentage data used for
Codice Galattico compliance checks (Impossible questions Q98-Q101).

Output: kb/dish_quantities.json
Format: {restaurant_name: {dish_name: {ingredient: pct_float}}}
"""

import json
import re
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

BLOG_DIR = Path("../../Dataset/knowledge_base/blogpost")
KB_DIR = Path("../kb")

# Ciascun blog è relativo ad un ristorante. Il nome del ristorante è quello che appare nel menù
BLOG_TO_RESTAURANT = {
    "blog_etere_del_gusto.html": "L'Etere del Gusto",
    "blog_sapore_del_dune.html": "Sapore del Dune",
}

# Regex costruite da Claude per riuscire ad estrarre i dati richiesti poi dalle domande senza ricorrere ad un LLM
DISH_PATTERN = re.compile(
    r'"([^"]+?)"[^.]*?(?:include|presenta|ha incluso|ha)',
    re.IGNORECASE,
)

PCT_PATTERN = re.compile(
    r'([\w\s\'àèéìòùÀÈÉÌÒÙ]+?)\s*\((?:[^)]*?)(\d+(?:[.,]\d+)?)\s*%[^)]*\)',
    re.IGNORECASE,
)


def parse_text_for_quantities(text: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    # Split in paragrafi
    paragraphs = re.split(r'\n+', text.strip())

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Cerco di i nomi dei piatti che sono tra doppi apici.
        # TODO: Attenzione, l'assunzione qui è molto fragile. Basta un nome di un piatto libero e viene perso
        dish_matches = re.findall(r'"([^"]+)"', para)
        if not dish_matches:
            continue

        # Nel paragrafo il primo nome tra doppi apici è quello del piatto
        dish_name = dish_matches[0].strip()

        # Ricerca delle percentuali di ingredienti
        pcts = PCT_PATTERN.findall(para)
        if not pcts:
            # Le regex le ha fatte Claude
            pcts = re.findall(
                r'(?:la\s+|il\s+|i\s+|le\s+|gli\s+)?'
                r'([A-Z][A-Za-zàèéìòùÀÈÉÌÒÙ\s\']+?)'
                r'\s*(?:\(.*?|\ballo?\b|\balla?\b\s*)'
                r'(\d+(?:[.,]\d+)?)\s*%',
                para,
            )

        # Estraggo ingredienti
        for ingredient_raw, pct_raw in pcts:
            ingredient = ingredient_raw.strip()
            ingredient = re.sub(
                r'^(la\s+|il\s+|i\s+|le\s+|gli\s+|un\s+|una\s+|del\s+|della\s+)',
                '',
                ingredient,
                flags=re.IGNORECASE,
            ).strip()
            if not ingredient or len(ingredient) < 3:
                continue

            pct = float(pct_raw.replace(',', '.'))
            if dish_name not in result:
                result[dish_name] = {}
            result[dish_name][ingredient] = pct

    return result


# Essendo i blog in html è stata utilizzata la libreria BeautifulSoup per riuscire a leggere il testo al suo interno
def parse_blog_bs4(html_path: Path) -> dict[str, dict[str, float]]:
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return parse_text_for_quantities(text)


# Questo in caso BeautifulSoup non sia installata
def parse_blog_regex(html_path: Path) -> dict[str, dict[str, float]]:
    html = html_path.read_text(encoding="utf-8")
    text = re.sub(r'<[^>]+>', '', html)
    text = text.replace('&#39;', "'").replace('&quot;', '"').replace('&amp;', '&')
    return parse_text_for_quantities(text)


def build_dish_quantities() -> None:
    KB_DIR.mkdir(exist_ok=True)

    all_quantities: dict[str, dict[str, dict[str, float]]] = {}

    for filename, restaurant in BLOG_TO_RESTAURANT.items():
        blog_path = BLOG_DIR / filename
        if not blog_path.exists():
            print(f"Attenzione: {blog_path} non trovato, skip")
            continue

        print(f"Parsing {filename} → {restaurant}")
        if HAS_BS4:
            dish_data = parse_blog_bs4(blog_path)
        else:
            print("BeautifulSoup non è disponibile, utilizzo delle regex per parsing")
            dish_data = parse_blog_regex(blog_path)

        all_quantities[restaurant] = dish_data
        for dish, ingredients in dish_data.items():
            print(f"  {dish}: {ingredients}")

    out_path = KB_DIR / "dish_quantities.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_quantities, f, indent=2, ensure_ascii=False)
    print(f"\nSalvati i dati sulle quantità dei piatti nel file {out_path}")


if __name__ == "__main__":
    build_dish_quantities()
