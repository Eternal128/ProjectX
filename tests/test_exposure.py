import pytest

from exposure.ngrams import item_queries, ngrams
from exposure.tiering import AMBIGUOUS, CLEAN, CONFIRMED, LIKELY, TierThresholds, assign_tier


def test_ngrams_cover_end_of_text():
    text = " ".join(str(i) for i in range(10))
    grams = ngrams(text, 4, stride=3)
    assert grams[0] == "0 1 2 3"
    assert grams[-1] == "6 7 8 9"
    assert ngrams("a b", 5) == ["a b"]
    assert ngrams("", 5) == []


def test_item_queries_qa_span_joins_question_and_answer():
    q = "Which planet is known as the red planet in our solar system today"
    out = item_queries(q, "Mars", long_n=13, short_n=8, stride=4)
    assert out["qa_span"][0].endswith("today Mars")
    assert len(out["qa_span"][0].split()) == 13
    assert out["long_answer"] == []


def test_tiers():
    th = TierThresholds(likely_min_similarity=0.85, clean_max_similarity=0.5)
    assert assign_tier(True, 0.1, th) == CONFIRMED
    assert assign_tier(False, 0.9, th) == LIKELY
    assert assign_tier(False, 0.3, th) == CLEAN
    assert assign_tier(False, None, th) == CLEAN
    assert assign_tier(False, 0.6, th) == AMBIGUOUS


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        TierThresholds(likely_min_similarity=0.4, clean_max_similarity=0.5)
