import re
from typing import Callable

_SYSTEM = (
    "Rispondi SOLO con i numeri [N] dei piatti che soddisfano la domanda. "
    "Se nessuno corrisponde scrivi 'nessuno'. Nessun altro testo."
)


def build_prompt(question: str, docs: list[tuple[int, str]]) -> str:
    doc_block = "\n\n".join(text for _, text in docs)
    return (
        f"DOMANDA: {question}\n\n"
        f"Leggi il campo 'Ingredienti:' e 'Tecniche:' di ogni piatto e indica i numeri [N] "
        f"dei piatti che soddisfano la domanda.\n\n"
        f"{doc_block}\n\n"
        f"Numeri dei piatti corrispondenti:"
    )


def parse_response(text: str) -> list[int]:
    """Extract seq_ids from the LLM response.

    Takes only the first line of the response to avoid extracting numbers
    from any explanation text the model might add.
    """
    if not text:
        return []
    first_line = text.strip().split("\n")[0].strip()
    if "nessuno" in first_line.lower():
        return []
    return [int(m) for m in re.findall(r"\b\d+\b", first_line)]


def llm_filter(
    call: Callable[[str, str], str],
    question: str,
    docs: list[tuple[int, str]],
) -> list[int]:
    """Return seq_ids of dishes matching the question.

    Args:
        call: callable(prompt, system) -> str  (from client.make_client)
        question: original full question (with exclusions)
        docs: list of (seq_id, doc_text) from retriever
    """
    if not docs:
        return []
    prompt = build_prompt(question, docs)
    raw = call(prompt, _SYSTEM)
    return parse_response(raw)
