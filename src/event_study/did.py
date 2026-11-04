"""Difference-in-differences on ``logp`` around each item's first exposure ``t0``.

Event time ``k`` counts checkpoints relative to exposure: ``k = 0`` is the first
checkpoint with ``step >= t0`` and ``k = -1`` the last one before it. For exposed
item ``i`` with matched clean items ``C(i)``:

    d_i(k) = [exp_i(k) - exp_i(-1)] - [mean_{c in C(i)} clean_c(k) - mean_{c in C(i)} clean_c(-1)]
    Effect(k) = mean_i d_i(k)

which equals the AGENTS.md formula with the clean mean taken over matched items.
Confidence intervals bootstrap over exposed items.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


def item_effects(df: pd.DataFrame, exposed: pd.DataFrame, matches: pd.DataFrame,
                 pre: int, post: int, reference: int = -1) -> pd.DataFrame:
    """Per-item ``d_i(k)`` for ``k`` in ``[-pre, post]`` for one model.

    ``df``: eval rows (model, benchmark, item_id, step, logp) for a single model.
    ``exposed``: (benchmark, item_id, t0). ``matches``: (benchmark, item_id, clean_item_id).
    Returns a frame indexed by (benchmark, item_id) with one column per ``k``.
    """
    if df["model"].nunique() != 1:
        raise ValueError("item_effects expects rows for a single model")
    wide = df.pivot_table(index=["benchmark", "item_id"], columns="step", values="logp")
    steps = np.asarray(wide.columns)
    ks = np.arange(-pre, post + 1)
    rows, index = [], []
    match_groups = matches.groupby(["benchmark", "item_id"])["clean_item_id"].apply(list)
    for bench, item, t0 in exposed[["benchmark", "item_id", "t0"]].itertuples(index=False):
        key = (bench, item)
        if key not in match_groups.index or key not in wide.index:
            continue
        pos0 = int(np.searchsorted(steps, t0, side="left"))      # first step >= t0
        pos = pos0 + ks
        ref = pos0 + reference
        if ref < 0 or ref >= len(steps):
            continue
        valid = (pos >= 0) & (pos < len(steps))
        clean_keys = [(bench, c) for c in match_groups[key] if (bench, c) in wide.index]
        if not clean_keys:
            continue
        exp_row = wide.loc[key].to_numpy()
        clean_mean = wide.loc[clean_keys].to_numpy().mean(axis=0)
        d = np.full(len(ks), np.nan)
        p = pos[valid]
        d[valid] = (exp_row[p] - exp_row[ref]) - (clean_mean[p] - clean_mean[ref])
        rows.append(d)
        index.append(key)
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["benchmark", "item_id"]),
                        columns=ks)


@dataclass
class EventStudyResult:
    effects: pd.DataFrame          # k, effect, ci_low, ci_high, n_items
    pretrend: float
    pretrend_ci: tuple[float, float]
    pretrend_flag: bool            # True: do not read this group's estimate causally


def _pre_mean(d: np.ndarray, ks: np.ndarray, reference: int) -> np.ndarray:
    pre_cols = ks < reference
    if not pre_cols.any():
        return np.full(d.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # items with no pre-period data
        return np.nanmean(d[:, pre_cols], axis=1)


def estimate(effects: pd.DataFrame, reps: int = 2000, level: float = 0.95, reference: int = -1,
             seed: int = 0) -> EventStudyResult:
    """Mean effect per event time with item-bootstrap CIs, plus the pre-trend test.

    The pre-trend statistic is the mean of ``d_i(k)`` over pre-period event times
    before the reference. It is flagged when its bootstrap CI excludes zero.
    """
    d = effects.to_numpy(dtype=float)
    ks = np.asarray(effects.columns, dtype=int)
    if d.shape[0] == 0:
        raise ValueError("no exposed items with usable matches")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.shape[0], size=(reps, d.shape[0]))
    alpha = (1 - level) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # event times with no items in a resample
        boot = np.stack([np.nanmean(d[i], axis=0) for i in idx])
        point = np.nanmean(d, axis=0)
        lo, hi = np.nanquantile(boot, [alpha, 1 - alpha], axis=0)
    eff = pd.DataFrame({"k": ks, "effect": point, "ci_low": lo, "ci_high": hi,
                        "n_items": np.sum(~np.isnan(d), axis=0)})

    pre_i = _pre_mean(d, ks, reference)
    ok = ~np.isnan(pre_i)
    if ok.sum() == 0:
        return EventStudyResult(eff, float("nan"), (float("nan"), float("nan")), False)
    pre_i = pre_i[ok]
    pre_boot = pre_i[rng.integers(0, len(pre_i), size=(reps, len(pre_i)))].mean(axis=1)
    p_lo, p_hi = np.quantile(pre_boot, [alpha, 1 - alpha])
    return EventStudyResult(eff, float(pre_i.mean()), (float(p_lo), float(p_hi)),
                            bool(p_lo > 0 or p_hi < 0))
