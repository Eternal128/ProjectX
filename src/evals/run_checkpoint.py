"""Evaluate Pythia checkpoints with per-sample logging and write parquet.

Example (one model, all configured checkpoints and kept benchmarks):

    python -m evals.run_checkpoint --config configs/evals.yaml --model pythia-410m

``tier``, ``t0`` and ``n`` are left empty here and joined from data/exposure later.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd

from common.config import load_config, provenance, write_parquet
from evals.extract import samples_to_rows
from evals.schema import EVAL_COLUMNS


def evaluate(hf_id: str, revision: str, task: str, *, batch_size="auto", dtype="float16",
             num_fewshot=0, limit=None, seed=1234) -> list[dict]:
    import lm_eval

    random.seed(seed)
    np.random.seed(seed)
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_id},revision={revision},dtype={dtype}",
        tasks=[task],
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        log_samples=True,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    return out["samples"][task]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/evals.yaml")
    ap.add_argument("--model", required=True, help="model name from configs/models.yaml")
    ap.add_argument("--benchmark", action="append", help="restrict to these benchmarks")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    models_cfg = load_config(cfg["models_config"])
    bench_cfg = load_config(cfg["benchmarks_config"])
    if not models_cfg.get("checkpoints_verified"):
        print("WARNING: checkpoint list not verified against Hugging Face (configs/models.yaml)")

    model = next(m for m in models_cfg["models"] if m["name"] == args.model)
    benches = [b for b in bench_cfg["benchmarks"] if b["status"] != "drop"]
    if args.benchmark:
        benches = [b for b in benches if b["name"] in args.benchmark]

    prov = provenance({"evals": cfg, "models": models_cfg, "benchmarks": bench_cfg}, args.config)
    for b in benches:
        for step in models_cfg["checkpoints"]:
            samples = evaluate(model["hf_id"], f"step{step}", b["lm_eval_task"],
                               batch_size=cfg["batch_size"], dtype=cfg["dtype"],
                               num_fewshot=cfg["num_fewshot"], limit=cfg["limit"], seed=cfg["seed"])
            rows = samples_to_rows(samples, model=model["name"], data_version=model["data_version"],
                                   step=step, benchmark=b["name"], output_type=b["output_type"])
            df = pd.DataFrame(rows, columns=EVAL_COLUMNS)
            path = f"{cfg['output_dir']}/{model['name']}/{b['name']}/step{step}.parquet"
            write_parquet(df, path, prov)
            print(f"{model['name']} {b['name']} step{step}: acc={df['y'].mean():.3f} -> {path}")


if __name__ == "__main__":
    main()
