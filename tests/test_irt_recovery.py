"""IRT parameter recovery on simulated data (AGENTS.md fitting order, steps 1-2)."""
import numpy as np
import pytest

from irt.counterfactual import accuracy_curves, exposure_share
from irt.data import simulate
from irt.fit import fit_map
from irt.models import STANDARD_2PL, ModelSpec


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


@pytest.mark.slow
def test_step1_map_shared_delta_recovers_parameters():
    data, truth = simulate(n_items=200, n_models=2, n_checkpoints=20, delta_0=1.0, seed=1)
    est = fit_map(data, ModelSpec(exposure=True), steps=3000, lr=0.03, seed=0)
    assert _corr(est["theta"], truth["theta"]) > 0.97
    assert _corr(est["b"], truth["b"]) > 0.9
    assert _corr(est["log_a"], truth["log_a"]) > 0.3
    assert abs(float(est["delta_0"]) - 1.0) < 0.3


@pytest.mark.slow
def test_step1_no_exposure_effect_gives_delta_near_zero():
    data, _ = simulate(n_items=200, n_models=2, n_checkpoints=20, delta_0=0.0, seed=2)
    est = fit_map(data, ModelSpec(exposure=True), steps=3000, lr=0.03, seed=0)
    assert abs(float(est["delta_0"])) < 0.3


@pytest.mark.slow
def test_step2_map_dose_and_ability_terms_recover_parameters():
    truth_d = dict(delta_0=0.5, delta_1=0.5, delta_2=-0.4)
    data, truth = simulate(n_items=400, n_models=3, n_checkpoints=20, frac_exposed=0.5, seed=3, **truth_d)
    est = fit_map(data, ModelSpec(exposure=True, dose=True, ability=True), steps=4000, lr=0.03, seed=0)
    assert _corr(est["theta"], truth["theta"]) > 0.97
    assert _corr(est["b"], truth["b"]) > 0.9
    for k, v in truth_d.items():
        assert abs(float(est[k]) - v) < 0.35, (k, float(est[k]), v)


@pytest.mark.slow
def test_standard_2pl_fits_and_counterfactual_share():
    data, _ = simulate(n_items=150, n_models=1, n_checkpoints=15, delta_0=1.5, seed=4)
    spec = ModelSpec(exposure=True)
    est = fit_map(data, spec, steps=2000, lr=0.03, seed=0)
    est_2pl = fit_map(data, STANDARD_2PL, steps=2000, lr=0.03, seed=0)
    assert "delta_0" not in est_2pl
    curves = accuracy_curves(est, data, spec)
    assert (curves["acc_e0"] <= curves["acc_predicted"] + 1e-9).all()  # positive delta only raises accuracy
    m = exposure_share(curves["acc_observed"].to_numpy(), curves["acc_e0"].to_numpy())
    assert 0.0 < m < 1.0


def test_exposure_share_formula():
    assert exposure_share(np.array([0.2, 0.6]), np.array([0.2, 0.5])) == pytest.approx(0.25)
    assert np.isnan(exposure_share(np.array([0.3, 0.3]), np.array([0.3, 0.2])))


def test_spec_rejects_terms_without_exposure():
    with pytest.raises(ValueError):
        ModelSpec(exposure=False, dose=True)
