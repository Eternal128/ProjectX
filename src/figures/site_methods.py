"""Method checks on simulated data, for the proposal and the project website.

    python -m figures.site_methods

Runs this project's own event-study and IRT code on simulated data with known, planted
effects, and writes site/data/methods.json. Nothing here is a result about Pythia: every
number comes from simulation and is labelled as such on the site.

1. Event study: a logp panel on the real 30-checkpoint schedule, with a planted jump at t0
   and no pre-trend; then the same panel with a pre-trend, which the test should flag.
2. Exposure IRT: MAP with a single shared delta recovers known item difficulties and delta.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config, provenance
from event_study.did import estimate, item_effects
from event_study.matching import base_difficulty, match_clean
from exposure.step_mapping import passes_timing_filter
from irt.data import simulate
from irt.fit import fit_map
from irt.models import ModelSpec


def simulated_panel(steps, total, jump, pretrend, n_exp=120, n_clean=360, seed=0):
    """logp for exposed and clean items: common upward trend, exposed items jump at t0."""
    rng = np.random.default_rng(seed)
    steps = np.asarray(steps)
    progress = np.log1p(steps) / np.log1p(total)
    eligible = [s for s in steps if passes_timing_filter(int(s), steps, total) and s <= 0.75 * total]
    rows, exposed = [], []
    for i in range(n_exp + n_clean):
        is_exp = i < n_exp
        base = rng.normal(-6, 1.2)
        slope = rng.normal(3.0, 0.4)
        t0 = int(rng.choice(eligible)) if is_exp else None
        for s, p in zip(steps, progress):
            lp = base + slope * p + rng.normal(0, 0.25)
            if is_exp:
                lp += jump * (s >= t0)
                if s < t0:
                    k_before = np.sum((steps < t0) & (steps >= s))  # checkpoints until t0
                    lp -= pretrend * k_before
            rows.append(("sim", "bench", f"i{i}", int(s), min(lp, 0.0), "confirmed" if is_exp else "clean", t0))
        if is_exp:
            exposed.append(("sim", "bench", f"i{i}", t0))
    df = pd.DataFrame(rows, columns=["model", "benchmark", "item_id", "step", "logp", "tier", "t0"])
    exp = pd.DataFrame(exposed, columns=["model", "benchmark", "item_id", "t0"])
    clean = df[df["tier"] == "clean"][["model", "benchmark", "item_id"]].drop_duplicates()
    return df, exp, clean


def group_curves(df, exposed, matches, pre, post, reference=-1):
    """Mean change in logp since the reference checkpoint, for exposed items and for their
    matched clean items, by event time (the two terms of the difference-in-differences)."""
    wide = df.pivot_table(index=["benchmark", "item_id"], columns="step", values="logp")
    steps = np.asarray(wide.columns)
    ks = np.arange(-pre, post + 1)
    groups = matches.groupby(["benchmark", "item_id"])["clean_item_id"].apply(list)
    exp_rows, clean_rows = [], []
    for bench, item, t0 in exposed[["benchmark", "item_id", "t0"]].itertuples(index=False):
        pos0 = int(np.searchsorted(steps, t0, side="left"))
        pos, ref = pos0 + ks, pos0 + reference
        valid = (pos >= 0) & (pos < len(steps))
        e = np.full(len(ks), np.nan)
        c = np.full(len(ks), np.nan)
        row = wide.loc[(bench, item)].to_numpy()
        cm = wide.loc[[(bench, x) for x in groups[(bench, item)]]].to_numpy().mean(axis=0)
        e[valid] = row[pos[valid]] - row[ref]
        c[valid] = cm[pos[valid]] - cm[ref]
        exp_rows.append(e)
        clean_rows.append(c)
    em, cm_ = np.nanmean(exp_rows, axis=0), np.nanmean(clean_rows, axis=0)
    return [[int(k), round(float(a), 3), round(float(b), 3)] for k, a, b in zip(ks, em, cm_)]


def run_event_study(cfg, steps, total, jump, pretrend, seed):
    df, exp, clean = simulated_panel(steps, total, jump, pretrend, seed=seed)
    diff = base_difficulty(df, cfg["matching"]["difficulty_max_step_fraction"] * total)
    matches = match_clean(diff, exp, clean, cfg["matching"]["clean_matches_per_item"])
    w = cfg["window"]
    d = item_effects(df, exp, matches, w["pre"], w["post"], w["reference"])
    res = estimate(d, reps=1000, level=cfg["bootstrap"]["level"], reference=w["reference"], seed=seed)
    return {
        "groups": group_curves(df, exp, matches, w["pre"], w["post"], w["reference"]),
        "planted_jump": jump, "planted_pretrend_per_checkpoint": pretrend, "n_exposed": int(len(d)),
        "effects": [[int(r.k), round(r.effect, 3), round(r.ci_low, 3), round(r.ci_high, 3), int(r.n_items)]
                    for r in res.effects.itertuples()],
        "pretrend": round(res.pretrend, 3), "pretrend_ci": [round(x, 3) for x in res.pretrend_ci],
        "pretrend_flag": res.pretrend_flag,
    }


def run_irt(seed=1):
    data, truth = simulate(n_items=200, n_models=2, n_checkpoints=20, delta_0=1.0, seed=seed)
    est = fit_map(data, ModelSpec(exposure=True), steps=3000, lr=0.03, seed=0)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    return {
        "planted_delta_0": 1.0, "recovered_delta_0": round(float(est["delta_0"]), 3),
        "difficulty": [[round(float(t), 3), round(float(e), 3)] for t, e in zip(truth["b"], est["b"])],
        "corr_difficulty": round(corr(truth["b"], est["b"]), 3),
        "corr_ability": round(corr(truth["theta"], est["theta"]), 3),
        "n_items": data.n_items, "n_abilities": data.n_abilities, "n_obs": int(len(data.y)),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event-study-config", default="configs/event_study.yaml")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--out", default="site/data/methods.json")
    args = ap.parse_args(argv)
    es_cfg = load_config(args.event_study_config)
    models = load_config(args.models_config)
    steps, total = models["checkpoints"], models["training"]["total_steps"]

    out = {
        "note": "Simulated data with known planted effects. Not results about Pythia.",
        "event_study": run_event_study(es_cfg, steps, total, jump=1.5, pretrend=0.0, seed=es_cfg["seed"]),
        "event_study_pretrend": run_event_study(es_cfg, steps, total, jump=1.5, pretrend=0.25, seed=es_cfg["seed"] + 1),
        "irt": run_irt(),
        "provenance": provenance({"event_study": es_cfg, "models": models}, args.event_study_config),
    }
    path = REPO_ROOT / args.out
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    es, pt, irt = out["event_study"], out["event_study_pretrend"], out["irt"]
    print("event study:", [e[:2] for e in es["effects"]], "pretrend flag", es["pretrend_flag"])
    print("with pretrend: flag", pt["pretrend_flag"], pt["pretrend"], pt["pretrend_ci"])
    print("irt:", irt["recovered_delta_0"], irt["corr_difficulty"], irt["corr_ability"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
