"""IRT models.

Exposure IRT, for item i, model m, checkpoint t (one ability index per (m, t)):

    P(y_imt = 1) = sigmoid( a_i * (theta_mt - b_i + delta_imt * E_imt) )
    delta_imt    = delta_0 + delta_1 * log(1 + n_it) + delta_2 * theta_mt

* theta_mt ~ Normal(0, 1) fixes the scale.
* b_i is base difficulty; a_i = exp(log_a_i) > 0 is discrimination.
* E_imt = 1 once model m has seen item i before checkpoint t.
* delta_2 (ability interaction) is the first term to drop if fitting fails.

With ``exposure=False`` this is the standard 2PL model used for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


@dataclass(frozen=True)
class Priors:
    theta: tuple[float, float] = (0.0, 1.0)
    b: tuple[float, float] = (0.0, 2.0)
    log_a: tuple[float, float] = (0.0, 0.5)
    delta: tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_config(cls, cfg: dict) -> "Priors":
        p = cfg["priors"]
        return cls(*[(p[k]["loc"], p[k]["scale"]) for k in ("theta", "b", "log_a", "delta")])


@dataclass(frozen=True)
class ModelSpec:
    exposure: bool = True
    dose: bool = False          # delta_1 * log(1 + n)
    ability: bool = False       # delta_2 * theta
    priors: Priors = field(default_factory=Priors)

    def __post_init__(self) -> None:
        if not self.exposure and (self.dose or self.ability):
            raise ValueError("dose/ability terms need exposure=True")


STANDARD_2PL = ModelSpec(exposure=False)


def linear_predictor(params: dict, item, ability, E, n, spec: ModelSpec):
    """Logit of P(y=1) from (possibly sampled) parameters."""
    theta = params["theta"][ability]
    eta = theta - params["b"][item]
    if spec.exposure:
        delta = params["delta_0"]
        if spec.dose:
            delta = delta + params["delta_1"] * jnp.log1p(n)
        if spec.ability:
            delta = delta + params["delta_2"] * theta
        eta = eta + delta * E
    return jnp.exp(params["log_a"][item]) * eta


def irt_model(item, ability, E, n, y=None, *, n_items: int, n_abilities: int, spec: ModelSpec):
    pr = spec.priors
    params = {
        "theta": numpyro.sample("theta", dist.Normal(*pr.theta).expand([n_abilities]).to_event(1)),
        "b": numpyro.sample("b", dist.Normal(*pr.b).expand([n_items]).to_event(1)),
        "log_a": numpyro.sample("log_a", dist.Normal(*pr.log_a).expand([n_items]).to_event(1)),
    }
    if spec.exposure:
        params["delta_0"] = numpyro.sample("delta_0", dist.Normal(*pr.delta))
        if spec.dose:
            params["delta_1"] = numpyro.sample("delta_1", dist.Normal(*pr.delta))
        if spec.ability:
            params["delta_2"] = numpyro.sample("delta_2", dist.Normal(*pr.delta))
    logits = linear_predictor(params, item, ability, E, n, spec)
    with numpyro.plate("obs", item.shape[0]):
        numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
