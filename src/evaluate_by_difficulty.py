"""
File creato solo per avere una panoramica sullo score per domande divise per difficoltà.

Run from src/:
    python evaluate_by_difficulty.py --submission ../submission.csv
"""

import sys

import click
import pandas as pd

from metrics.jaccard_similarity import score

SOLUTION_PATH = "../Dataset/ground_truth/ground_truth_mapped.csv"
QUESTIONS_PATH = "../Dataset/domande.csv"

DIFFICULTY_ORDER = ["Easy", "Medium", "Hard", "Impossible"]


def _jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union else 0.0


def _parse_ids(value) -> set[int]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    if isinstance(value, (int, float)):
        return {int(value)}
    return {int(x.strip()) for x in str(value).split(",") if x.strip()}


@click.command()
@click.option(
    "--submission",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help="Path to the submission CSV file to evaluate.",
)
@click.option(
    "--row-id-column",
    default="row_id",
    show_default=True,
    help="Column used to align rows.",
)
def main(submission: str, row_id_column: str) -> None:
    """Jaccard similarity score broken down by question difficulty."""
    solution_df = pd.read_csv(SOLUTION_PATH)
    submission_df = pd.read_csv(submission)
    questions_df = pd.read_csv(QUESTIONS_PATH)

    # row_id is 1-based position in domande.csv
    questions_df[row_id_column] = range(1, len(questions_df) + 1)

    # Global score via the canonical metric
    try:
        global_score = score(
            solution=solution_df,
            submission=submission_df,
            row_id_column_name=row_id_column,
        )
    except ValueError as e:
        click.echo(f"Invalid submission: {e}", err=True)
        sys.exit(1)

    # Per-row Jaccard for the breakdown
    merged = (
        solution_df[[row_id_column, "result"]]
        .rename(columns={"result": "truth"})
        .merge(
            submission_df[[row_id_column, "result"]].rename(columns={"result": "pred"}),
            on=row_id_column,
        )
        .merge(
            questions_df[[row_id_column, "difficoltà"]],
            on=row_id_column,
        )
    )

    merged["jaccard"] = merged.apply(
        lambda r: _jaccard(_parse_ids(r["truth"]), _parse_ids(r["pred"])),
        axis=1,
    )

    # Print results
    click.echo(f"\n{'Difficulty':<12}  {'Questions':>9}  {'Jaccard':>8}")
    click.echo("-" * 35)

    for diff in DIFFICULTY_ORDER:
        subset = merged[merged["difficoltà"] == diff]
        if subset.empty:
            continue
        diff_score = subset["jaccard"].mean() * 100
        click.echo(f"{diff:<12}  {len(subset):>9}  {diff_score:>7.2f}%")

    click.echo("-" * 35)
    click.echo(f"{'GLOBAL':<12}  {len(merged):>9}  {global_score:>7.2f}%")
    click.echo()


if __name__ == "__main__":
    main()
