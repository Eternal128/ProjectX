"""Match each exposed item to clean items from the same benchmark with similar
base difficulty."""
from __future__ import annotations

import numpy as np
import pandas as pd


def base_difficulty(df: pd.DataFrame, max_step: float) -> pd.DataFrame:
    """Mean ``logp`` per (model, benchmark, item_id) over checkpoints with
    ``step < max_step``. Higher means easier."""
    early = df[df["step"] < max_step]
    if early.empty:
        raise ValueError(f"no checkpoints before step {max_step} to measure base difficulty")
    return (early.groupby(["model", "benchmark", "item_id"], as_index=False)["logp"].mean()
            .rename(columns={"logp": "difficulty"}))


def match_clean(difficulty: pd.DataFrame, exposed: pd.DataFrame, clean: pd.DataFrame, k: int) -> pd.DataFrame:
    """Nearest-difficulty matches, with replacement across exposed items.

    ``exposed`` and ``clean`` have columns (model, benchmark, item_id). Returns
    (model, benchmark, item_id, clean_item_id, distance), ``k`` rows per exposed
    item (fewer if the benchmark has fewer clean items).
    """
    d = difficulty.set_index(["model", "benchmark", "item_id"])["difficulty"]
    out = []
    for (model, bench), grp in exposed.groupby(["model", "benchmark"]):
        pool = clean[(clean["model"] == model) & (clean["benchmark"] == bench)]["item_id"].unique()
        if len(pool) == 0:
            continue
        pool_d = np.array([d[(model, bench, c)] for c in pool])
        order = np.argsort(pool_d, kind="stable")
        pool, pool_d = pool[order], pool_d[order]
        for item in grp["item_id"].unique():
            dist = np.abs(pool_d - d[(model, bench, item)])
            nearest = np.argsort(dist, kind="stable")[:k]
            out += [(model, bench, item, pool[j], float(dist[j])) for j in nearest]
    return pd.DataFrame(out, columns=["model", "benchmark", "item_id", "clean_item_id", "distance"])
