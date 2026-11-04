"""Download EleutherAI's published Pythia evaluation results and summarise them.

    python -m evals.pythia_published

The Pythia repository (Apache-2.0) ships zero-shot lm-evaluation-harness results for every
model at 27 checkpoints. They cover four of our candidate benchmarks (LAMBADA, PIQA, SciQ,
ARC-Easy) and MMLU, but not TriviaQA or SQuAD. They are aggregate accuracies from EleutherAI's
own harness run, not our per-item evaluations: they are used for the Phase 0 above-chance check
and to describe the benchmarks, never as inputs to the IRT model or the event study.

Writes:
  data/raw/pythia_evals/<model>/<file>.json   raw files, pinned to one commit (gitignored)
  data/evals/pythia_published.parquet          long table with provenance (gitignored)
  site/data/benchmarks.json                    small summary for the project website
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT, load_config, provenance, write_parquet

REPO = "EleutherAI/pythia"
COMMIT = "a19eecb807ec2c79a39ebf18108816e6ffffc1d5"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/evals/pythia-v1"
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000] + list(range(13000, 143001, 10000))
TASKS = {  # published task name -> our benchmark name
    "lambada_openai": "lambada",
    "piqa": "piqa",
    "sciq": "sciq",
    "arc_easy": "arc_easy",
}
MMLU_PREFIX = "hendrycksTest-"


def model_short(name: str) -> str:
    return name.removeprefix("pythia-")


def raw_path(model: str, step: int) -> Path:
    short = model_short(model)
    return REPO_ROOT / "data" / "raw" / "pythia_evals" / model / f"{short}_step{step}.json"


def download(model: str, step: int) -> dict:
    path = raw_path(model, step)
    if not path.exists():
        short = model_short(model)
        url = f"{RAW}/{model}/zero-shot/{short}_step{step}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return json.loads(path.read_text())


def rows_for(model: str, data_version: str, step: int, results: dict) -> list[dict]:
    rows = []
    for task, bench in TASKS.items():
        r = results[task]
        rows.append({"model": model, "data_version": data_version, "step": step, "benchmark": bench,
                     "acc": r["acc"], "acc_stderr": r["acc_stderr"]})
    mmlu = [v for k, v in results.items() if k.startswith(MMLU_PREFIX)]
    acc = sum(v["acc"] for v in mmlu) / len(mmlu)
    se = math.sqrt(sum(v["acc_stderr"] ** 2 for v in mmlu)) / len(mmlu)
    rows.append({"model": model, "data_version": data_version, "step": step, "benchmark": "mmlu",
                 "acc": acc, "acc_stderr": se})
    return rows


def above_chance(df: pd.DataFrame, chance: dict[str, float], margin: float, alpha: float) -> pd.DataFrame:
    """configs/benchmarks.yaml rule: accuracy beats chance by `margin` and a one-sided z-test against
    chance has p < alpha (normal approximation with the published standard error)."""
    from scipy.stats import norm

    final = df[df["step"] == df["step"].max()].copy()
    final["chance"] = final["benchmark"].map(chance)
    final["margin"] = final["acc"] - final["chance"]
    final["p_value"] = norm.sf(final["margin"] / final["acc_stderr"])
    final["passes"] = (final["margin"] >= margin) & (final["p_value"] < alpha)
    return final


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--benchmarks-config", default="configs/benchmarks.yaml")
    ap.add_argument("--out", default="site/data/benchmarks.json")
    args = ap.parse_args(argv)
    models_cfg = load_config(args.models_config)
    bench_cfg = load_config(args.benchmarks_config)

    rows = []
    for m in models_cfg["models"]:
        for step in STEPS:
            rows += rows_for(m["name"], m["data_version"], step, download(m["name"], step)["results"])
    df = pd.DataFrame(rows)
    cfg = {"models": models_cfg, "benchmarks": bench_cfg, "source": {"repo": REPO, "commit": COMMIT}}
    prov = provenance(cfg, args.benchmarks_config)
    write_parquet(df, "data/evals/pythia_published.parquet", prov)

    chance = {b["name"]: b["chance"] for b in bench_cfg["benchmarks"]}
    chance["mmlu"] = 0.25
    rule = bench_cfg["above_chance"]
    final = above_chance(df, chance, rule["min_margin"], rule["alpha"])

    names = {"lambada": "LAMBADA", "piqa": "PIQA", "sciq": "SciQ", "arc_easy": "ARC-Easy", "mmlu": "MMLU"}
    out = {
        "source": {
            "description": "Zero-shot accuracy published by EleutherAI for the Pythia suite (lm-evaluation-harness).",
            "repo": f"https://github.com/{REPO}/tree/{COMMIT}/evals/pythia-v1",
            "commit": COMMIT, "licence": "Apache-2.0",
            "note": "Aggregate accuracy from EleutherAI's harness run, not this project's per-item evaluation. "
                    "TriviaQA and SQuAD are not in the published results.",
        },
        "rule": {"min_margin": rule["min_margin"], "alpha": rule["alpha"]},
        "steps": STEPS,
        "models": [{"name": m["name"], "data_version": m["data_version"]} for m in models_cfg["models"]],
        "benchmarks": [{"key": k, "name": names[k], "chance": chance[k], "in_study": k != "mmlu"} for k in names],
        "series": {
            m: {b: [[int(r.step), round(r.acc, 4), round(r.acc_stderr, 4)] for r in g.sort_values("step").itertuples()]
                for b, g in gm.groupby("benchmark")}
            for m, gm in df.groupby("model")
        },
        "final": [
            {"model": r.model, "benchmark": r.benchmark, "acc": round(r.acc, 4), "stderr": round(r.acc_stderr, 4),
             "chance": r.chance, "margin": round(r.margin, 4), "passes": bool(r.passes)}
            for r in final.itertuples()
        ],
        "provenance": prov,
    }
    path = REPO_ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(final[["model", "benchmark", "acc", "chance", "margin", "p_value", "passes"]].to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
