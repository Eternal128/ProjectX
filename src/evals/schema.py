"""Schema of the per-item, per-checkpoint evaluation table.

Inputs must not be mixed: IRT uses binary ``y`` only, the event study uses
continuous ``logp`` only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVAL_COLUMNS = ["model", "data_version", "step", "benchmark", "item_id", "y", "logp", "tier", "t0", "n"]
DATA_VERSIONS = {"standard", "deduped"}
TIER_VALUES = {"confirmed", "likely", "clean", "ambiguous", None}
KEY = ["model", "step", "benchmark", "item_id"]


def validate_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if ``df`` does not follow the evaluation schema."""
    missing = [c for c in EVAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not set(df["data_version"].unique()) <= DATA_VERSIONS:
        raise ValueError(f"data_version must be one of {DATA_VERSIONS}")
    if not df["y"].isin([0, 1]).all():
        raise ValueError("y must be binary 0/1")
    if not np.isfinite(df["logp"]).all() or (df["logp"] > 1e-6).any():
        raise ValueError("logp must be finite and <= 0")
    tiers = set(df["tier"].where(df["tier"].notna(), None).unique())
    if not tiers <= TIER_VALUES:
        raise ValueError(f"unknown tiers: {tiers - TIER_VALUES}")
    if df.duplicated(KEY).any():
        raise ValueError(f"duplicate rows for key {KEY}")
    return df
