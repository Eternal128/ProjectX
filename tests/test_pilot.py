from evals.benchmarks import items_arc_easy, items_piqa, items_sciq
from exposure.search_items import clean_text, tier_of


def test_sciq_prompt_matches_harness_format():
    row = {"question": "What gas do plants take in?", "distractor1": "oxygen", "distractor2": "helium",
           "distractor3": "neon", "correct_answer": "carbon dioxide", "support": "  Plants use CO2."}
    it = items_sciq([row])[0]
    assert it["prompt"] == "Plants use CO2.\nQuestion: What gas do plants take in?\nAnswer:"
    assert it["choices"][it["answer"]] == " carbon dioxide"


def test_arc_answer_key_maps_to_choice_index():
    row = {"id": "x", "answerKey": "3", "question": {"stem": "Pick three.", "choices": [
        {"text": "one", "label": "1"}, {"text": "two", "label": "2"}, {"text": "three", "label": "3"}]}}
    it = items_arc_easy([row])[0]
    assert it["answer"] == 2 and it["answer_text"] == "three"
    assert it["prompt"] == "Question: Pick three.\nAnswer:"


def test_piqa_uses_label():
    it = items_piqa([{"goal": "Open a jar", "sol1": "twist the lid", "sol2": "shout at it", "label": 0}])[0]
    assert it["choices"] == [" twist the lid", " shout at it"] and it["answer"] == 0


def test_tiers():
    long_q = "which of these is the main source of energy for most food webs on earth"
    assert tier_of(long_q, q_count=3, qa_count=1, min_words=8) == "confirmed"
    assert tier_of(long_q, q_count=3, qa_count=0, min_words=8) == "question_only"
    assert tier_of(long_q, q_count=0, qa_count=0, min_words=8) == "clean"
    assert tier_of("what is it", q_count=500, qa_count=9, min_words=8) == "ambiguous"


def test_clean_text_keeps_query_operators_out_of_items():
    assert clean_text("salt AND  pepper OR sugar") == "salt and pepper or sugar"
