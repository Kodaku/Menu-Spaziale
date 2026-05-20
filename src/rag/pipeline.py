import csv
import json
import sys
from pathlib import Path

import click
import dotenv

HERE = Path(__file__).parent
CORPUS_PATH = HERE / "corpus.json"
SEQ_TO_NAME_PATH = HERE / "seq_to_name.json"
KB_PATH = HERE / ".." / "kb" / "dishes.json"
QUESTIONS_PATH = HERE / ".." / ".." / "Dataset" / "domande.csv"
MAPPING_PATH = HERE / ".." / ".." / "Dataset" / "ground_truth" / "dish_mapping.json"
DEFAULT_OUTPUT = HERE / ".." / ".." / "submission_rag_llama3.1-8b.csv"

# Add src/ to path so we can import id_mapper
sys.path.insert(0, str(HERE / ".."))


@click.command()
@click.option(
    "--backend",
    default="groq",
    show_default=True,
    help="LLM backend: 'groq' or 'ollama:<model>'",
)
@click.option("--k", default=20, show_default=True, help="BM25 top-k candidates")
@click.option(
    "--difficulty",
    default=None,
    help="Only process questions of this difficulty (Easy/Medium/Hard/Impossible)",
)
@click.option(
    "--output",
    default=str(DEFAULT_OUTPUT),
    show_default=True,
    help="Output CSV path",
)
@click.option("--verbose", is_flag=True, help="Print retrieved doc count and raw LLM response")
def main(backend: str, k: int, difficulty: str | None, output: str, verbose: bool) -> None:
    dotenv.load_dotenv(HERE / ".." / ".." / ".env")

    # Lazy imports so errors surface clearly
    from build_index import build
    from client import make_client
    from id_mapper import DishMapper
    from llm_filter import llm_filter
    from retriever import BM25Index

    # Build index if corpus is missing
    if not CORPUS_PATH.exists() or not SEQ_TO_NAME_PATH.exists():
        click.echo("Building index...")
        corpus, seq_to_name = build()
    else:
        with open(CORPUS_PATH, encoding="utf-8") as f:
            corpus = json.load(f)
        with open(SEQ_TO_NAME_PATH, encoding="utf-8") as f:
            seq_to_name = {int(k): v for k, v in json.load(f).items()}
        click.echo(f"Loaded index: {len(corpus)} documents")

    index = BM25Index(corpus)
    call = make_client(backend)
    mapper = DishMapper(str(MAPPING_PATH))

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-processed row_ids
    cache: dict[int, str] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for cached_row in csv.DictReader(f):
                if cached_row["result"].strip():  # skip empty rows so they get reprocessed
                    cache[int(cached_row["row_id"])] = cached_row["result"]
        click.echo(f"Resuming: {len(cache)} questions already cached\n")

    results: list[dict] = []
    skipped = 0

    for i, row in enumerate(rows, 1):
        question = row["domanda"]
        diff = row.get("difficoltà", "")

        if i in cache:
            results.append({"row_id": i, "result": cache[i]})
            continue

        if difficulty and diff != difficulty:
            results.append({"row_id": i, "result": ""})
            skipped += 1
            continue

        click.echo(f"[{i:03d}/{len(rows)}] ({diff}) {question[:60]}...", nl=False)

        try:
            docs = index.retrieve(question, k=k)
            if verbose:
                click.echo(f"\n  retrieved {len(docs)} docs")

            seq_ids = llm_filter(call, question, docs)
            if verbose:
                click.echo(f"  llm returned seq_ids: {seq_ids}")

            # seq_ids → dish names → dish IDs
            # Guard: only use seq_ids that were actually in the retrieved set
            valid_seq = {sid for sid, _ in docs}
            dish_names = [seq_to_name[sid] for sid in seq_ids if sid in valid_seq and sid in seq_to_name]
            ids = mapper.map_names(dish_names)
            result_str = ",".join(str(d) for d in ids)
            click.echo(f"  -> {len(ids)} piatti")
        except Exception as e:
            click.echo(f"  ERROR: {e}")
            result_str = ""

        results.append({"row_id": i, "result": result_str})

        # Save incrementally
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["row_id", "result"])
            writer.writeheader()
            writer.writerows(results)

    answered = len(results) - skipped - len(cache)
    click.echo(
        f"\nDone. {answered} new + {len(cache)} cached = {len(results)} total -> {output_path}"
    )


if __name__ == "__main__":
    main()
