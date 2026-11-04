"""Write site/data/results.json for the project website, reading only from data/ and configs/.

    python -m figures.site_data

Every result block stays "pending" until its outputs exist in data/results/. Run this only
after the analysis code and thresholds are frozen (AGENTS.md conventions): it publishes
whatever is in data/results/ to the site.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT, load_config, provenance, read_provenance

SITE = REPO_ROOT / "site"
PENDING = {"status": "pending", "summary": None, "figure": None}


def _copy_figure(src: Path, name: str) -> str | None:
    """Copy an SVG figure into site/figures/ and return its site-relative path."""
    if not src.exists():
        return None
    dest = SITE / "figures" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return f"figures/{name}"


def event_study_block(es_cfg: dict) -> dict:
    out = REPO_ROOT / es_cfg["output_dir"]
    eff_path, pre_path = out / "effects.parquet", out / "pretrend.parquet"
    if not eff_path.exists():
        return dict(PENDING)
    eff = pd.read_parquet(eff_path)
    pre = pd.read_parquet(pre_path)
    eff = eff[(eff["stratum"] == "all") & (eff["k"] >= 0)]
    parts = []
    for model, g in eff.groupby("model"):
        flag = pre[(pre["model"] == model) & (pre["stratum"] == "all")]["pretrend_flag"]
        note = " (pre-trend: not causal)" if len(flag) and bool(flag.iloc[0]) else ""
        parts.append(f"{model}: {g['effect'].mean():+.2f}{note}")
    fig = REPO_ROOT / Path(es_cfg["figure"]["output"]).with_suffix(".svg")
    return {
        "status": "available",
        "summary": "Mean change in log-probability after exposure, seen minus matched unseen. " + "; ".join(parts),
        "figure": _copy_figure(fig, "event_study.svg"),
        "figure_alt": "Event study: change in log-probability around the step each question was read.",
        "provenance": read_provenance(eff_path),
    }


def counterfactual_block(irt_cfg: dict) -> dict:
    summary_path = REPO_ROOT / irt_cfg["output_dir"] / "summary.json"
    if not summary_path.exists():
        return dict(PENDING)
    s = json.loads(summary_path.read_text())
    shares = "; ".join(f"{m}: {v:.0%}" for m, v in s["exposure_share_M"].items())
    fig = REPO_ROOT / Path(irt_cfg["figure"]["output"]).with_suffix(".svg")
    return {
        "status": "available",
        "summary": f"Share of accuracy gain explained by exposure. {shares}",
        "figure": _copy_figure(fig, "counterfactual.svg"),
        "figure_alt": "Observed accuracy over training next to the curve with exposure removed.",
        "provenance": s.get("provenance"),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--exposure-config", default="configs/exposure.yaml")
    ap.add_argument("--event-study-config", default="configs/event_study.yaml")
    ap.add_argument("--irt-config", default="configs/irt.yaml")
    ap.add_argument("--out", default="site/data/results.json")
    args = ap.parse_args(argv)

    models = load_config(args.models_config)
    exposure = load_config(args.exposure_config)
    es_cfg = load_config(args.event_study_config)
    irt_cfg = load_config(args.irt_config)

    blocks = {
        "event_study": event_study_block(es_cfg),
        "counterfactual": counterfactual_block(irt_cfg),
        "ability": dict(PENDING),     # H1/H4: no site export yet
        "transfer": dict(PENDING),    # H5: paraphrase check not built yet
    }
    data = {
        "status": "available" if any(b["status"] == "available" for b in blocks.values()) else "pending",
        "design": {
            "total_steps": models["training"]["total_steps"],
            "checkpoints": models["checkpoints"],
            "timing_filter": exposure["timing_filter"],
        },
        "blocks": blocks,
        "provenance": provenance({"models": models, "exposure": exposure, "event_study": es_cfg, "irt": irt_cfg}),
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {out} ({data['status']})")


if __name__ == "__main__":
    main()
