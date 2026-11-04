"""Run the event study from data/evals and write data/results/event_study/.

    python -m event_study.run --config configs/event_study.yaml

Per model and per t0 stratum. Exposed = confirmed items passing the timing filter.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.config import load_config, provenance, write_parquet
from event_study.did import estimate, item_effects
from event_study.matching import base_difficulty, match_clean
from exposure.step_mapping import passes_timing_filter


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/event_study.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    models_cfg = load_config(cfg["models_config"])
    tf = load_config(cfg["exposure_config"])["timing_filter"]
    total = models_cfg["training"]["total_steps"]

    df = pd.read_parquet(cfg["evals_path"], columns=["model", "step", "benchmark", "item_id", "logp", "tier", "t0"])
    items = df.drop_duplicates(["model", "benchmark", "item_id"])[["model", "benchmark", "item_id", "tier", "t0"]]
    steps_by_model = df.groupby("model")["step"].apply(lambda s: sorted(s.unique()))
    exposed = items[(items["tier"] == "confirmed") & items.apply(
        lambda r: passes_timing_filter(None if pd.isna(r["t0"]) else int(r["t0"]), steps_by_model[r["model"]],
                                       total, tf["min_checkpoints_before_t0"], tf["t0_min_fraction"],
                                       tf["t0_max_fraction"]), axis=1)]
    clean = items[items["tier"] == "clean"]

    diff = base_difficulty(df, cfg["matching"]["difficulty_max_step_fraction"] * total)
    matches = match_clean(diff, exposed, clean, cfg["matching"]["clean_matches_per_item"])

    w, bs = cfg["window"], cfg["bootstrap"]
    effects, summary = [], []
    for model, exp_m in exposed.groupby("model"):
        exp_m = exp_m.copy()
        exp_m["t0_stratum"] = pd.qcut(exp_m["t0"], cfg["t0_strata"], labels=False, duplicates="drop")
        strata = [("all", exp_m)] + [(f"t0_q{int(q)}", g) for q, g in exp_m.groupby("t0_stratum")]
        for name, grp in strata:
            d = item_effects(df[df["model"] == model], grp, matches[matches["model"] == model],
                             w["pre"], w["post"], w["reference"])
            res = estimate(d, bs["reps"], bs["level"], w["reference"], cfg["seed"])
            effects.append(res.effects.assign(model=model, stratum=name))
            summary.append({"model": model, "stratum": name, "n_items": len(d),
                            "pretrend": res.pretrend, "pretrend_ci_low": res.pretrend_ci[0],
                            "pretrend_ci_high": res.pretrend_ci[1], "pretrend_flag": res.pretrend_flag})

    prov = provenance(cfg, args.config)
    write_parquet(pd.concat(effects), f"{cfg['output_dir']}/effects.parquet", prov)
    write_parquet(pd.DataFrame(summary), f"{cfg['output_dir']}/pretrend.parquet", prov)
    write_parquet(matches, f"{cfg['output_dir']}/matches.parquet", prov)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
