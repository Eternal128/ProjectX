"""Summarise the CPU pilot for the website: site/data/pilot.json.

    python -m figures.site_pilot

Reads data/evals/pilot/<model>/step*.parquet (scores) and, when present,
data/exposure/pilot_tiers.parquet (Pile search). Writes:
  - our accuracy per benchmark and checkpoint, next to EleutherAI's published accuracy
    at the same checkpoint (a check that our scoring matches theirs);
  - per exposure group (confirmed / question_only / clean): accuracy and mean
    log-probability of the right answer at each checkpoint, with 95% bootstrap
    intervals over items;
  - how many items fall in each group, and a few confirmed examples.

This is descriptive: it compares seen and unseen items over training. It does not
estimate the step each item was first read (that needs the preshuffled training data),
so it is not the event study.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, provenance
from evals.benchmarks import load_items

GROUPS = ["confirmed", "question_only", "clean"]


def boot_ci(x: np.ndarray, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--out", default="site/data/pilot.json")
    args = ap.parse_args(argv)
    files = sorted((REPO_ROOT / "data" / "evals" / "pilot" / args.model).glob("step*.parquet"))
    if not files:
        raise SystemExit("no pilot scores yet: run python -m evals.score_cpu first")
    df = pd.concat([pd.read_parquet(f) for f in files])
    steps = sorted(df["step"].unique().tolist())

    published = json.loads((REPO_ROOT / "site" / "data" / "benchmarks.json").read_text())
    pub = {b: {p[0]: p[1] for p in pts} for b, pts in published["series"][f"pythia-{args.model.split('-')[-1]}"].items()} \
        if f"pythia-{args.model.split('-')[-1]}" in published["series"] else {}

    out = {"model": args.model, "steps": steps, "n_items": int(df.groupby("step").size().iloc[0]),
           "accuracy": {}, "groups": {}, "tier_counts": {}, "examples": []}
    for b, g in df.groupby("benchmark"):
        acc = g.groupby("step")["y"].mean()
        out["accuracy"][b] = [[int(s), round(float(acc[s]), 4), pub.get(b, {}).get(int(s))] for s in steps]

    tiers_path = REPO_ROOT / "data" / "exposure" / "pilot_tiers.parquet"
    if tiers_path.exists():
        tiers = pd.read_parquet(tiers_path)[["benchmark", "item_id", "tier", "q_count", "qa_count"]]
        out["tier_counts"] = {b: {k: int(v) for k, v in g["tier"].value_counts().items()} for b, g in tiers.groupby("benchmark")}
        m = df.merge(tiers, on=["benchmark", "item_id"], how="inner")
        for b, gb in list(m.groupby("benchmark")) + [("all", m)]:
            out["groups"][b] = {}
            for t in GROUPS:
                gt = gb[gb["tier"] == t]
                if gt.empty:
                    continue
                series = []
                for s in steps:
                    gs = gt[gt["step"] == s]
                    y, lp = gs["y"].to_numpy(float), gs["logp"].to_numpy(float)
                    series.append([int(s), round(float(y.mean()), 4), *[round(v, 4) for v in boot_ci(y)],
                                   round(float(lp.mean()), 4), *[round(v, 4) for v in boot_ci(lp)]])
                out["groups"][b][t] = {"n": int(gt["item_id"].nunique()), "series": series}
        items = {(it["benchmark"], it["item_id"]): it for it in load_items()}
        conf = tiers[tiers["tier"] == "confirmed"].sort_values("qa_count", ascending=False)
        for r in conf.head(6).itertuples():
            it = items[(r.benchmark, r.item_id)]
            out["examples"].append({"benchmark": r.benchmark, "question": it["question"], "answer": it["answer_text"],
                                    "copies": int(r.qa_count)})
    out["provenance"] = provenance({"model": args.model, "steps": steps})
    path = REPO_ROOT / args.out
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
