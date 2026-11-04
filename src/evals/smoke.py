"""Phase 0 smoke test: evaluate one Pythia checkpoint on one benchmark.

    python -m evals.smoke --model EleutherAI/pythia-410m --revision step143000 --task arc_easy --limit 50
"""
from __future__ import annotations

import argparse

import pandas as pd

from evals.extract import samples_to_rows
from evals.run_checkpoint import evaluate


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--task", default="arc_easy")
    ap.add_argument("--output-type", default="multiple_choice")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args(argv)

    samples = evaluate(args.model, args.revision, args.task, limit=args.limit)
    rows = samples_to_rows(samples, model=args.model, data_version="standard",
                           step=int(args.revision.removeprefix("step")), benchmark=args.task,
                           output_type=args.output_type)
    df = pd.DataFrame(rows)
    print(df[["item_id", "y", "logp"]].head())
    print(f"n={len(df)}  accuracy={df['y'].mean():.3f}  mean logp={df['logp'].mean():.3f}")


if __name__ == "__main__":
    main()
