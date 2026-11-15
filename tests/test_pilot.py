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


def test_coverage_windows_take_first_and_last_13_words():
    from exposure.probe_coverage import windows

    words = [f"w{i}" for i in range(20)]
    w = windows(" ".join(words))
    assert w["first"].split() == words[:13] and w["last"].split() == words[-13:]
    assert windows("too short to have a window") == {}


def test_coverage_wilson_interval_brackets_the_share():
    from exposure.probe_coverage import wilson

    lo, hi = wilson(2, 100)
    assert 0 < lo < 0.02 < hi < 0.08
    assert wilson(0, 50)[0] == 0.0


def test_coverage_summary_counts_an_item_once_across_windows():
    import pandas as pd
    from exposure.probe_coverage import summarise

    df = pd.DataFrame([
        {"benchmark": "sciq", "item_id": "a", "check": "window_first", "count": 2},
        {"benchmark": "sciq", "item_id": "a", "check": "window_last", "count": 5},
        {"benchmark": "sciq", "item_id": "b", "check": "window_first", "count": 0},
        {"benchmark": "triviaqa", "item_id": "q1", "check": "question", "count": 3},
        {"benchmark": "triviaqa", "item_id": "q1", "check": "question_and_answer", "count": 1},
        {"benchmark": "triviaqa", "item_id": "q2", "check": "question", "count": 0},
    ])
    s = summarise(df).set_index(["benchmark", "check"])
    assert s.loc[("sciq", "13-word window found"), "found"] == 1 and s.loc[("sciq", "13-word window found"), "items"] == 2
    assert s.loc[("triviaqa", "question and answer found"), "share"] == 0.5
