"""Build IRT inputs from the evaluation table and simulate data for recovery tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IRTData:
    item: np.ndarray        # int32 item index
    ability: np.ndarray     # int32 (model, checkpoint) index
    E: np.ndarray           # float32 exposure indicator
    n: np.ndarray           # float32 running copy count
    y: np.ndarray           # int32 binary correctness
    item_ids: list
    ability_ids: list       # list of (model, step)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_abilities(self) -> int:
        return len(self.ability_ids)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"item": self.item, "ability": self.ability, "E": self.E, "n": self.n, "y": self.y}


def from_eval_frame(df: pd.DataFrame) -> IRTData:
    """IRT inputs from the evaluation table. Uses ``y`` only, never ``logp``.

    ``E`` is 1 when ``t0`` is known and ``step >= t0``; ``n`` defaults to 0.
    """
    item_key = df["benchmark"].astype(str) + "::" + df["item_id"].astype(str)
    item_codes, item_ids = pd.factorize(item_key, sort=True)
    ab = list(zip(df["model"], df["step"]))
    ab_codes, ab_ids = pd.factorize(pd.Series(ab), sort=True)
    t0 = pd.to_numeric(df["t0"], errors="coerce")
    E = (t0.notna() & (df["step"] >= t0)).to_numpy(np.float32)
    n = pd.to_numeric(df["n"], errors="coerce").fillna(0).to_numpy(np.float32) * E
    return IRTData(item_codes.astype(np.int32), ab_codes.astype(np.int32), E, n,
                   df["y"].to_numpy(np.int32), list(item_ids), list(ab_ids))


def simulate(
    n_items: int = 150,
    n_models: int = 2,
    n_checkpoints: int = 20,
    frac_exposed: float = 0.4,
    delta_0: float = 1.0,
    delta_1: float = 0.0,
    delta_2: float = 0.0,
    seed: int = 0,
) -> tuple[IRTData, dict[str, np.ndarray]]:
    """Simulate exposure-IRT data with known parameters.

    Ability rises over checkpoints for every model; a fraction of items gets a
    first-exposure checkpoint after the first three, with copies accumulating
    afterwards.
    """
    rng = np.random.default_rng(seed)
    progress = np.linspace(-1.5, 1.5, n_checkpoints)
    theta = (progress[None, :] + rng.normal(0, 0.3, (n_models, 1))
             + rng.normal(0, 0.1, (n_models, n_checkpoints))).ravel()
    b = rng.normal(0, 1.0, n_items)
    log_a = rng.normal(0, 0.3, n_items)

    exposed = rng.random(n_items) < frac_exposed
    t0_idx = np.where(exposed, rng.integers(3, n_checkpoints - 2, n_items), n_checkpoints + 1)
    rate = rng.poisson(1.5, n_items) + 1

    ii, mm, tt = np.meshgrid(np.arange(n_items), np.arange(n_models), np.arange(n_checkpoints), indexing="ij")
    ii, mm, tt = ii.ravel(), mm.ravel(), tt.ravel()
    ability = mm * n_checkpoints + tt
    E = (tt >= t0_idx[ii]).astype(np.float32)
    n = (E * rate[ii] * (tt - t0_idx[ii] + 1)).astype(np.float32)

    th = theta[ability]
    delta = delta_0 + delta_1 * np.log1p(n) + delta_2 * th
    logits = np.exp(log_a[ii]) * (th - b[ii] + delta * E)
    y = (rng.random(len(ii)) < 1 / (1 + np.exp(-logits))).astype(np.int32)

    data = IRTData(ii.astype(np.int32), ability.astype(np.int32), E, n, y,
                   list(range(n_items)), [(m, t) for m in range(n_models) for t in range(n_checkpoints)])
    truth = {"theta": theta, "b": b, "log_a": log_a, "delta_0": np.float64(delta_0),
             "delta_1": np.float64(delta_1), "delta_2": np.float64(delta_2)}
    return data, truth
