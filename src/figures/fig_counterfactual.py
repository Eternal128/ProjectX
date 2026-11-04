"""Second figure: observed accuracy over training next to the exposure-removed
counterfactual curve, gap shaded.

    python -m figures.fig_counterfactual --config configs/irt.yaml
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
    ap.add_argument("--config", default="configs/irt.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    curves = pd.read_parquet(REPO_ROOT / cfg["output_dir"] / "accuracy_curves.parquet")
    models = sorted(curves["model"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        c = curves[curves["model"] == model].sort_values("step")
        ax.plot(c["step"], c["acc_observed"], label="observed")
        ax.plot(c["step"], c["acc_e0"], ls="--", label="exposure removed (E=0)")
        ax.fill_between(c["step"], c["acc_e0"], c["acc_observed"], alpha=0.2, lw=0)
        ax.set_xscale("log")
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("training step")
    axes[0][0].set_ylabel("accuracy")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = REPO_ROOT / cfg["figure"]["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
