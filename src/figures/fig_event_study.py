"""Main figure: event study of logp around first exposure, exposed vs matched clean.

    python -m figures.fig_event_study --config configs/event_study.yaml

Model-free: plots the difference-in-differences effect with item-bootstrap CIs.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.config import REPO_ROOT, load_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/event_study.yaml")
    ap.add_argument("--stratum", default="all")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    eff = pd.read_parquet(REPO_ROOT / cfg["output_dir"] / "effects.parquet")
    pre = pd.read_parquet(REPO_ROOT / cfg["output_dir"] / "pretrend.parquet")
    eff = eff[eff["stratum"] == args.stratum]
    models = sorted(eff["model"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        e = eff[eff["model"] == model].sort_values("k")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(-0.5, color="0.6", lw=0.8, ls="--")
        ax.fill_between(e["k"], e["ci_low"], e["ci_high"], alpha=0.25, lw=0)
        ax.plot(e["k"], e["effect"], marker="o", ms=3)
        flag = pre[(pre["model"] == model) & (pre["stratum"] == args.stratum)]["pretrend_flag"]
        title = model + ("  (pre-trend!)" if len(flag) and bool(flag.iloc[0]) else "")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("checkpoints since exposure (k)")
    axes[0][0].set_ylabel("Δ log p(correct), exposed − clean")
    fig.tight_layout()
    out = REPO_ROOT / cfg["figure"]["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
