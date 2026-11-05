import math

import pandas as pd

from evals.pythia_published import above_chance, rows_for


def test_rows_for_maps_tasks_and_averages_mmlu():
    results = {
        "lambada_openai": {"acc": 0.5, "acc_stderr": 0.01},
        "piqa": {"acc": 0.7, "acc_stderr": 0.01},
        "sciq": {"acc": 0.8, "acc_stderr": 0.01},
        "arc_easy": {"acc": 0.6, "acc_stderr": 0.01},
        "hendrycksTest-a": {"acc": 0.2, "acc_stderr": 0.03},
        "hendrycksTest-b": {"acc": 0.3, "acc_stderr": 0.04},
    }
    rows = {r["benchmark"]: r for r in rows_for("pythia-410m", "standard", 143000, results)}
    assert set(rows) == {"lambada", "piqa", "sciq", "arc_easy", "mmlu"}
    assert rows["mmlu"]["acc"] == 0.25
    assert math.isclose(rows["mmlu"]["acc_stderr"], 0.025)


def test_above_chance_rule_uses_margin_and_significance():
    df = pd.DataFrame({
        "model": "m", "step": 143000, "benchmark": ["piqa", "mmlu", "sciq"],
        "acc": [0.70, 0.27, 0.32], "acc_stderr": [0.01, 0.005, 0.05],
    })
    out = above_chance(df, {"piqa": 0.5, "mmlu": 0.25, "sciq": 0.25}, margin=0.05, alpha=0.01).set_index("benchmark")
    assert out.loc["piqa", "passes"]
    assert not out.loc["mmlu", "passes"]   # significant but under the 5-point margin
    assert not out.loc["sciq", "passes"]   # over the margin but not significant
