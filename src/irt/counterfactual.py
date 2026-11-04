"""Exposure share of progress from the exposure-removed counterfactual.

    M = (dAcc_observed - dAcc_E0) / dAcc_observed

where dAcc_E0 is the predicted accuracy change with every E set to 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from irt.data import IRTData
from irt.models import ModelSpec, linear_predictor


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def accuracy_curves(params: dict, data: IRTData, spec: ModelSpec) -> pd.DataFrame:
    """Per (model, step): observed accuracy, model-predicted accuracy, and the
    predicted accuracy with exposure removed (E = 0)."""
    p = {k: np.asarray(v) for k, v in params.items()}
    a = data.arrays()
    pred = _sigmoid(np.asarray(linear_predictor(p, a["item"], a["ability"], a["E"], a["n"], spec)))
    zero = np.zeros_like(a["E"])
    pred_e0 = _sigmoid(np.asarray(linear_predictor(p, a["item"], a["ability"], zero, zero, spec)))
    df = pd.DataFrame({"ability": a["ability"], "y": a["y"], "pred": pred, "pred_e0": pred_e0})
    out = df.groupby("ability", as_index=False).agg(acc_observed=("y", "mean"),
                                                   acc_predicted=("pred", "mean"),
                                                   acc_e0=("pred_e0", "mean"))
    out["model"] = [data.ability_ids[i][0] for i in out["ability"]]
    out["step"] = [data.ability_ids[i][1] for i in out["ability"]]
    return out.drop(columns="ability").sort_values(["model", "step"]).reset_index(drop=True)


def exposure_share(acc_observed: np.ndarray, acc_e0: np.ndarray) -> float:
    """M for one accuracy curve ordered by step, using first and last checkpoints."""
    d_obs = acc_observed[-1] - acc_observed[0]
    d_e0 = acc_e0[-1] - acc_e0[0]
    if np.isclose(d_obs, 0.0):
        return float("nan")
    return float((d_obs - d_e0) / d_obs)
