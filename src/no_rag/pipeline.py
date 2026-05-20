"""
End-to-end pipeline: domande.csv → query parser → filter engine → ID mapper → submission.csv

Run from src/:
    python pipeline.py                          # all questions (groq backend)
    python pipeline.py --backend ollama:qwen2.5:7b
    python pipeline.py --difficulty Easy        # only Easy questions
    python pipeline.py --output ../my_run.csv   # custom output path
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import click
import dotenv
from openai import OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filter_engine import filter_dishes, load_db
from id_mapper import DishMapper
from query_parser import QueryParser
from constants import API_KEY_ENV_NAME, GROQ_MODEL_LLAMA_VERSATILE

QUESTIONS_PATH = Path("../../Dataset/domande.csv")
MAPPING_PATH = Path("../../Dataset/ground_truth/dish_mapping.json")
KB_PATH = Path("../kb/dishes.json")
DEFAULT_OUTPUT = Path("../../submission_llama3.3-70b.csv")

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Qui si da la possibilità anche di utilizzare Ollama. Ma i modelli utilizzabili in locale (leggeri e veloci, e questo dipende molto dal pc che si utilizza) con le risorse a disposizione non garantivano un risultato affidabile.
# La dimostrazione di questa affermazione è stata fatta con Claude code che ad un certo punto era entrato in un loop e in un bias di ragionamento tale per cui:
# 1. Continuava ad aggiustare i prompt di estrazione delle keyword dalle domande.
# 2. Se un prompt funzionava per la domanda X passava alla Y, ma a questo punto il prompt andava riaggiustato e quindi poi non andava più bene per la X, vista l'alta suscettibilità all'allucinazione (risposte non conformi a quanto richiesto inventando la struttura dell'output richiesto).
# 3. Di conseguenza si tornava al punto 1.
def _make_client(backend: str) -> Callable:
    if backend == "groq":
        api_key = os.environ.get(API_KEY_ENV_NAME)
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        oai = OpenAI(base_url=_GROQ_BASE_URL, api_key=api_key)
        model = GROQ_MODEL_LLAMA_VERSATILE
    elif backend.startswith("ollama:"):
        model = backend[len("ollama:"):]
        oai = OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")
    else:
        raise ValueError(f"Unknown backend {backend!r}. Use 'groq' or 'ollama:<model>'.")

    def call(prompt: str, system: str, max_tokens: int = 512) -> str:
        delay = 60
        for attempt in range(3):
            try:
                kwargs: dict = dict(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                kwargs["response_format"] = {"type": "json_object"}
                resp = oai.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                if attempt == 2:
                    raise
                wait = delay * 3
                click.echo(f"\n  rate limit, retry {attempt+2}/3 in {wait}s: {e}")
                time.sleep(wait)
                delay *= 2
            except Exception as e:
                if attempt == 2:
                    raise
                click.echo(f"\n  retry {attempt+2}/3 in {delay}s: {e}")
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("all retries exhausted")

    return call


@click.command()
@click.option(
    "--backend",
    default="groq",
    show_default=True,
    help="LLM backend: 'groq' or 'ollama:<model>'",
)
@click.option("--difficulty", default=None, help="Filter by difficulty (Easy/Medium/Hard/Impossible)")
@click.option("--output", default=str(DEFAULT_OUTPUT), show_default=True, help="Output CSV path")
@click.option("--questions", default=str(QUESTIONS_PATH), show_default=True, help="Questions CSV path")
@click.option("--verbose", is_flag=True, help="Print filter JSON for each question")
def main(backend: str, difficulty: str | None, output: str, questions: str, verbose: bool) -> None:
    dotenv.load_dotenv()

    call = _make_client(backend)
    parser = QueryParser(call, KB_PATH)
    db = load_db(KB_PATH)
    mapper = DishMapper(str(MAPPING_PATH))

    with open(questions, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Carico i risultati già pre-processati
    cache: dict[int, str] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for cached_row in csv.DictReader(f):
                if cached_row["result"].strip():
                    cache[int(cached_row["row_id"])] = cached_row["result"]
        click.echo(f"Ricomincio da: {len(cache)} domande già in cache {output_path}\n")

    # Ripopolo i risultati delle domande già processate. Perché può succedere di dover interrompere il processo per limiti raggiunti nell'utilizzo dell'LLM e quindi rilanciarlo. Di default le domande senza risposta sono sempre riprocessate.
    results: list[dict] = [
        {"row_id": i + 1, "result": cache.get(i + 1, "")} for i in range(len(rows))
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "result"])
        writer.writeheader()
        writer.writerows(results)

    skipped = 0

    for i, row in enumerate(rows, 1):
        question = row["domanda"]
        diff = row.get("difficoltà", "")

        if i in cache:
            continue

        if difficulty and diff != difficulty:
            skipped += 1
            continue

        click.echo(f"[{i:03d}/{len(rows)}] ({diff}) {question[:65]}...", nl=False)

        try:
            # Estrazione dei filtri
            filters = parser.parse(question)
            if verbose:
                click.echo(f"\n  filtri: {json.dumps(filters, ensure_ascii=False)}")
            dish_names = filter_dishes(db, filters)
            ids = mapper.map_names(dish_names)
            result_str = ",".join(str(d) for d in ids)
            click.echo(f"  -> {len(ids)} piatti")
        except Exception as e:
            click.echo(f"  ERROR: {e}")
            result_str = ""

        results[i - 1]["result"] = result_str

        # Save full results after each newly processed question
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["row_id", "result"])
            writer.writeheader()
            writer.writerows(results)

    answered = len(rows) - skipped - len(cache)
    click.echo(f"\nFatto. Risposte {answered} nuove + {len(cache)} cached = {len(rows)} totali -> {output_path}")


if __name__ == "__main__":
    main()
