"""
BM25 retriever over the dish corpus.
"""

import re

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    # Strip punctuation from boundaries so "cioccorane?" == "cioccorane"
    # but keep tokens like "latte+" intact (mid-word punctuation preserved)
    tokens = text.lower().split()
    return [t.strip(".,;:!?\"'()[]") for t in tokens if t.strip(".,;:!?\"'()[]")]


def _strip_negations(question: str) -> str:
    """Remove negation markers so BM25 focuses on positive terms.

    Strips patterns like "senza X", "evitando X", "non usa X", etc.
    leaving only the ingredient/technique names as positive signals.
    """
    # Remove the negation keyword; the noun phrase that follows stays,
    # which is intentional: BM25 will still retrieve docs mentioning it,
    # but without the negation keyword artificially boosting those docs.
    q = re.sub(
        r"\b(senza|evitando|escludendo|non\s+(?:contiene|usa|utilizza|impiega|coinvolge|includendo|impiegando|usando|utilizzando|facendo\s+uso\s+di))\b",
        "",
        question,
        flags=re.IGNORECASE,
    )
    return " ".join(q.split())

# La parte di retrieve (di RAG) è svolta non con un db vettoriale ma con un retriever che, data la domanda e il corpus, stima la pertinenza di un documento con la query. Siccome ci possono essere query espresse in negativo (es. non contiene X) le parole di negazione vengono rimosse mantenendo la parola che viene negata.
# Una volta che la query va in input all'LLM insieme ai K documenti presi in esame l'LLM provvederà a capire che i documenti dati in input che contengono gli ingredienti da escludere non dovranno essere considerati.
# TODO: questa soluzione è più fragile rispetto a quella con db per i seguenti motivi:
# 1. Si delega molto all'LLM, in particolare la facoltà di capire che un ingrediente nel contesto aumentato va escluso. Se la keyword di negazione non viene enfatizzata può essere che il modello la ignori.
# 2. Se si chiede di escludere un ingrediente o una tecnica e di considerare tutti i piatti senza, ad esempio, le Chochobo Wings verranno presi in esame i soli documenti che menzionano questo ingrediente che poi saranno scartati dall'LLM.
# 3. Se non si adotta la tecnica dell'esclusione delle negazioni risulta particolarmente inefficiente adottare un RAG siccome il contesto rischierebbe di essere aumentato con tutti gli ingredienti
class BM25Index:
    def __init__(self, corpus: list[str]):
        self._corpus = corpus
        self._index = BM25Okapi([_tokenize(doc) for doc in corpus])

    def retrieve(self, question: str, k: int = 50) -> list[tuple[int, str]]:
        """Return top-k (seq_id, doc_text) pairs for the given question.

        seq_id is 1-indexed (matches seq_to_name.json keys).
        """
        query = _strip_negations(question)
        scores = self._index.get_scores(_tokenize(query))

        # argsort descending, take top k
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i + 1, self._corpus[i]) for i in ranked]
