"""Fitting: MAP, SVI and NUTS, in the order required by AGENTS.md.

1. MAP with a single shared delta on simulated data (tests/test_irt_recovery.py).
2. Add dose and ability terms, MAP, recovery test again.
3. SVI on real data for approximate uncertainty.
4. NUTS on a random subset of items, only to confirm the SVI intervals.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.optim import Adam

from irt.data import IRTData
from irt.models import ModelSpec, irt_model


def _kwargs(data: IRTData, spec: ModelSpec) -> dict:
    a = data.arrays()
    return {
        "item": jnp.asarray(a["item"]), "ability": jnp.asarray(a["ability"]),
        "E": jnp.asarray(a["E"]), "n": jnp.asarray(a["n"]), "y": jnp.asarray(a["y"]),
        "n_items": data.n_items, "n_abilities": data.n_abilities, "spec": spec,
    }


def _run_svi(guide_cls, data, spec, steps, lr, seed):
    guide = guide_cls(irt_model)
    svi = SVI(irt_model, guide, Adam(lr), Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(seed), steps, progress_bar=False, **_kwargs(data, spec))
    return guide, result


def fit_map(data: IRTData, spec: ModelSpec, steps: int = 4000, lr: float = 0.02, seed: int = 0) -> dict:
    """MAP point estimates (AutoDelta guide)."""
    guide, result = _run_svi(AutoDelta, data, spec, steps, lr, seed)
    params = guide.median(result.params)
    out = {k: np.asarray(v) for k, v in params.items()}
    out["final_loss"] = float(result.losses[-1])
    return out


def fit_svi(data: IRTData, spec: ModelSpec, steps: int = 20000, lr: float = 0.005,
            num_samples: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """Approximate posterior samples from a mean-field Normal guide."""
    guide, result = _run_svi(AutoNormal, data, spec, steps, lr, seed)
    kw = _kwargs(data, spec)
    kw["y"] = None
    pred = Predictive(guide, params=result.params, num_samples=num_samples)
    samples = pred(jax.random.PRNGKey(seed + 1), **kw)
    return {k: np.asarray(v) for k, v in samples.items()}


def fit_nuts(data: IRTData, spec: ModelSpec, num_warmup: int = 1000, num_samples: int = 1000,
             num_chains: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    mcmc = MCMC(NUTS(irt_model), num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), **_kwargs(data, spec))
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


def item_subset(data: IRTData, size: int, seed: int = 0) -> IRTData:
    """Random subset of items (re-indexed) for the NUTS confirmation step."""
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(data.n_items, size=min(size, data.n_items), replace=False))
    remap = -np.ones(data.n_items, dtype=np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    mask = remap[data.item] >= 0
    return IRTData(remap[data.item[mask]], data.ability[mask], data.E[mask], data.n[mask],
                   data.y[mask], [data.item_ids[k] for k in keep], data.ability_ids)
