"""Summarise the CPU pilot for the website: site/data/pilot.json.

    python -m figures.site_pilot

Reads data/evals/pilot/<model>/step*.parquet (scores) and, when present,
data/exposure/pilot_tiers.parquet (Pile search). Writes:
  - our accuracy per benchmark and checkpoint, next to EleutherAI's published accuracy
    at the same checkpoint (a check that our scoring matches theirs);
  - per exposure group (confirmed / question_only / clean): accuracy and mean
    log-probability of the right answer at each checkpoint, with 95% bootstrap
    intervals over items;
  - how many items fall in each group, and a few confirmed examples;
  - when data/exposure/probe_coverage.parquet exists, the share of items found by the
    wider coverage probe (13-word windows, LAMBADA and TriviaQA samples), with 95% intervals.

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
    lam_files = sorted((REPO_ROOT / "data" / "evals" / "pilot" / args.model / "lambada").glob("step*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files + lam_files])
    # each benchmark keeps the checkpoints it has been scored at (runs can be partway through);
    # `steps` is their union, and every series row carries its own step
    steps = sorted(df["step"].unique().tolist())

    published = json.loads((REPO_ROOT / "site" / "data" / "benchmarks.json").read_text())
    pub = {b: {p[0]: p[1] for p in pts} for b, pts in published["series"][f"pythia-{args.model.split('-')[-1]}"].items()} \
        if f"pythia-{args.model.split('-')[-1]}" in published["series"] else {}

    out = {"model": args.model, "steps": steps, "n_items": int(df.groupby("step").size().iloc[0]),
           "accuracy": {}, "groups": {}, "tier_counts": {}, "examples": []}
    for b, g in df.groupby("benchmark"):
        acc = g.groupby("step")["y"].mean()
        out["accuracy"][b] = [[int(s), round(float(acc[s]), 4), pub.get(b, {}).get(int(s))] for s in sorted(acc.index)]

    tiers_path = REPO_ROOT / "data" / "exposure" / "pilot_tiers.parquet"
    if tiers_path.exists():
        tiers = pd.read_parquet(tiers_path)[["benchmark", "item_id", "tier", "q_count", "qa_count"]]
        cov_path = REPO_ROOT / "data" / "exposure" / "probe_coverage.parquet"
        if lam_files and cov_path.exists():
            # LAMBADA: the probe's 13-word window ends in the answer word, so a hit means the
            # passage and its answer were in the Pile ("confirmed"); no hit means "clean"
            cov = pd.read_parquet(cov_path)
            lam = cov[cov["benchmark"] == "lambada"].groupby("item_id")["count"].max().reset_index()
            tiers = pd.concat([tiers, pd.DataFrame({
                "benchmark": "lambada", "item_id": "lambada-" + lam["item_id"].astype(str),
                "tier": ["confirmed" if c > 0 else "clean" for c in lam["count"]],
                "q_count": lam["count"].astype(int), "qa_count": lam["count"].astype(int)})], ignore_index=True)
        out["tier_counts"] = {b: {k: int(v) for k, v in g["tier"].value_counts().items()} for b, g in tiers.groupby("benchmark")}
        m = df.drop(columns=["tier", "t0", "n"], errors="ignore").merge(tiers, on=["benchmark", "item_id"], how="inner")
        pilot_three = m[m["benchmark"] != "lambada"]
        common = set.intersection(*(set(g["step"]) for _, g in pilot_three.groupby("benchmark"))) if len(pilot_three) else set()
        for b, gb in list(m.groupby("benchmark")) + [("all", pilot_three[pilot_three["step"].isin(common)])]:
            out["groups"][b] = {}
            for t in GROUPS:
                gt = gb[gb["tier"] == t]
                if gt.empty:
                    continue
                series = []
                for s in sorted(gt["step"].unique()):
                    gs = gt[gt["step"] == s]
                    y, lp = gs["y"].to_numpy(float), gs["logp"].to_numpy(float)
                    series.append([int(s), round(float(y.mean()), 4), *[round(v, 4) for v in boot_ci(y)],
                                   round(float(lp.mean()), 4), *[round(v, 4) for v in boot_ci(lp)]])
                out["groups"][b][t] = {"n": int(gt["item_id"].nunique()), "series": series}
        all_items = load_items()
        lam_items = REPO_ROOT / "data" / "raw" / "items_lambada.jsonl"
        if lam_items.exists():
            all_items += load_items(lam_items)
        items = {(it["benchmark"], it["item_id"]): it for it in all_items}
        conf = tiers[(tiers["tier"] == "confirmed") & (tiers["benchmark"] != "lambada")].sort_values("qa_count", ascending=False)
        for r in conf.head(6).itertuples():
            it = items[(r.benchmark, r.item_id)]
            out["examples"].append({"benchmark": r.benchmark, "question": it["question"], "answer": it["answer_text"],
                                    "copies": int(r.qa_count)})
    lam_src = "lambada_all" if (REPO_ROOT / "data" / "exposure" / "lambada_all_first_exposure.parquet").exists() else "lambada"
    for source in ("pilot", lam_src):
        fe = REPO_ROOT / "data" / "exposure" / f"{source}_first_exposure.parquet"
        if fe.exists():
            f = pd.read_parquet(fe)
            out.setdefault("dates", {})["lambada" if source == "lambada_all" else source] = {
                "items": int(len(f)), "dated": int(f["t0"].notna().sum()), "pass_timing": int(f["passes_timing"].sum()) if "passes_timing" in f else None,
                "t0": sorted(int(t) for t in f["t0"].dropna()),
            }
    cov_path = REPO_ROOT / "data" / "exposure" / "probe_coverage.parquet"
    if cov_path.exists():
        from exposure.probe_coverage import summarise

        out["coverage"] = summarise(pd.read_parquet(cov_path)).to_dict(orient="records")
    out["provenance"] = provenance({"model": args.model, "steps": steps})
    path = REPO_ROOT / args.out
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
