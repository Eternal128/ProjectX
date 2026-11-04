import numpy as np
import pandas as pd

from event_study.did import estimate, item_effects
from event_study.matching import base_difficulty, match_clean

STEPS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]


def _panel(jump=2.0, pretrend=0.0, n_exp=60, n_clean=120, seed=0):
    """Synthetic logp panel: common upward trend, exposed items jump at t0."""
    rng = np.random.default_rng(seed)
    rows, exposed = [], []
    for i in range(n_exp + n_clean):
        is_exp = i < n_exp
        base = rng.normal(-5, 1)
        t0 = int(rng.choice(STEPS[5:9])) if is_exp else None
        for s in STEPS:
            lp = base + 0.002 * s + rng.normal(0, 0.1)
            if is_exp:
                lp += jump * (s >= t0) + pretrend * (s - t0) / 100 * (s < t0)
            rows.append(("m", "bench", f"i{i}", s, lp, "confirmed" if is_exp else "clean", t0))
        if is_exp:
            exposed.append(("m", "bench", f"i{i}", t0))
    df = pd.DataFrame(rows, columns=["model", "benchmark", "item_id", "step", "logp", "tier", "t0"])
    exp = pd.DataFrame(exposed, columns=["model", "benchmark", "item_id", "t0"])
    clean = df[df["tier"] == "clean"][["model", "benchmark", "item_id"]].drop_duplicates()
    return df, exp, clean


def _run(df, exp, clean, k=3):
    diff = base_difficulty(df, max_step=300)
    m = match_clean(diff, exp, clean, k=k)
    d = item_effects(df, exp, m, pre=4, post=3)
    return m, d, estimate(d, reps=300, seed=0)


def test_did_recovers_jump_without_pretrend():
    df, exp, clean = _panel(jump=2.0)
    m, d, res = _run(df, exp, clean)
    assert len(m) == 3 * len(exp)
    eff = res.effects.set_index("k")["effect"]
    assert abs(eff[-1]) < 1e-9
    assert all(abs(eff[k] - 2.0) < 0.15 for k in range(0, 4))
    assert all(abs(eff[k]) < 0.15 for k in (-4, -3, -2))
    assert not res.pretrend_flag


def test_null_effect_ci_covers_zero():
    df, exp, clean = _panel(jump=0.0, seed=1)
    _, _, res = _run(df, exp, clean)
    e = res.effects.set_index("k")
    assert ((e["ci_low"] <= 0) & (e["ci_high"] >= 0)).loc[0:3].mean() >= 0.75


def test_pretrend_is_flagged():
    df, exp, clean = _panel(jump=2.0, pretrend=0.5, seed=2)
    _, _, res = _run(df, exp, clean)
    assert res.pretrend_flag
    assert res.pretrend < 0


def test_matching_prefers_similar_difficulty():
    diff = pd.DataFrame({"model": "m", "benchmark": "b", "item_id": ["e", "c1", "c2", "c3"],
                         "difficulty": [-2.0, -2.1, -5.0, -1.8]})
    exp = pd.DataFrame({"model": ["m"], "benchmark": ["b"], "item_id": ["e"]})
    clean = pd.DataFrame({"model": "m", "benchmark": "b", "item_id": ["c1", "c2", "c3"]})
    m = match_clean(diff, exp, clean, k=2)
    assert set(m["clean_item_id"]) == {"c1", "c3"}
