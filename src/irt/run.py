"""Fit IRT models on data/evals and write data/results/irt/.

    python -m irt.run --config configs/irt.yaml

Fits the configured exposure-IRT stage and the standard 2PL for comparison,
then writes accuracy curves with the exposure-removed counterfactual.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config, provenance, write_parquet
from irt.counterfactual import accuracy_curves, exposure_share
from irt.data import from_eval_frame
from irt.fit import fit_map, fit_svi
from irt.models import ModelSpec, Priors


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/irt.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    df = pd.read_parquet(cfg["evals_path"], columns=["model", "step", "benchmark", "item_id", "y", "t0", "n"])
    data = from_eval_frame(df)

    priors = Priors.from_config(cfg)
    spec = ModelSpec(exposure=True, dose=cfg["terms"]["dose"], ability=cfg["terms"]["ability"], priors=priors)
    base = ModelSpec(exposure=False, priors=priors)  # standard 2PL for comparison

    m = cfg["map"]
    if cfg["stage"] == "svi":
        s = cfg["svi"]
        samples = fit_svi(data, spec, s["steps"], s["learning_rate"], s["posterior_samples"], cfg["seed"])
        params = {k: v.mean(axis=0) for k, v in samples.items() if k != "y"}
        intervals = {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in samples.items() if k.startswith("delta")}
    else:
        params = fit_map(data, spec, m["steps"], m["learning_rate"], cfg["seed"])
        intervals = {}
    params_2pl = fit_map(data, base, m["steps"], m["learning_rate"], cfg["seed"])

    curves = accuracy_curves(params, data, spec)
    curves_2pl = accuracy_curves(params_2pl, data, base).rename(columns={"acc_predicted": "acc_predicted_2pl"})
    curves = curves.merge(curves_2pl[["model", "step", "acc_predicted_2pl"]], on=["model", "step"])
    shares = {mdl: exposure_share(g["acc_observed"].to_numpy(), g["acc_e0"].to_numpy())
              for mdl, g in curves.groupby("model")}

    prov = provenance(cfg, args.config)
    out = cfg["output_dir"]
    write_parquet(curves, f"{out}/accuracy_curves.parquet", prov)
    theta = pd.DataFrame({"model": [a[0] for a in data.ability_ids], "step": [a[1] for a in data.ability_ids],
                          "theta": params["theta"], "theta_2pl": params_2pl["theta"]})
    write_parquet(theta, f"{out}/abilities.parquet", prov)
    summary = {"provenance": prov, "stage": cfg["stage"], "exposure_share_M": shares,
               "deltas": {k: float(v) for k, v in params.items() if k.startswith("delta")},
               "delta_intervals_95": intervals}
    (REPO_ROOT / out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
