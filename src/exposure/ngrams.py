"""N-gram extraction for infini-gram queries.

Tokenization here is a simple word-level split used to choose query spans. The
infini-gram index tokenizes the query text itself, so the span length in index
tokens can differ slightly; record both in the candidate table.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"\S+")


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def ngrams(text: str, n: int, stride: int = 1) -> list[str]:
    """All word n-grams of length ``n`` taken every ``stride`` words.

    The final n-gram is always included so the end of the text is covered.
    Texts shorter than ``n`` words return the whole text as a single query.
    """
    if n < 1 or stride < 1:
        raise ValueError("n and stride must be >= 1")
    toks = words(text)
    if len(toks) <= n:
        return [" ".join(toks)] if toks else []
    starts = list(range(0, len(toks) - n + 1, stride))
    if starts[-1] != len(toks) - n:
        starts.append(len(toks) - n)
    return [" ".join(toks[s : s + n]) for s in starts]


def item_queries(question: str, answer: str, long_n: int, short_n: int, stride: int) -> dict[str, list[str]]:
    """Queries for one benchmark item.

    ``long_question``/``long_answer`` drive the confirmed tier; ``qa_span`` joins the
    end of the question to the answer so a single exact match covers both;
    ``short`` widens the candidate set for the semantic check.
    """
    q_words, a_words = words(question), words(answer)
    tail = q_words[-max(long_n - len(a_words), 1):]
    return {
        "long_question": ngrams(question, long_n, stride),
        "long_answer": ngrams(answer, long_n, stride) if len(a_words) >= long_n else [],
        "qa_span": [" ".join(tail + a_words)] if a_words else [],
        "short": ngrams(question + " " + answer, short_n, stride),
    }
