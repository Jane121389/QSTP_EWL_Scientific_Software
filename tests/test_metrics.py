
import numpy as np
from qstp_ewl.core import ExperimentPoint
from qstp_ewl.metrics import (
    wilson_interval,
    estimate_violation_volume,
    perturb_point,
    critical_threshold,
)

def test_wilson_interval_is_valid():
    low, high = wilson_interval(10, 100)
    assert 0 <= low <= 0.1 <= high <= 1

def test_volume_bounds():
    result = estimate_violation_volume("original", "standard", samples=100, seed=1)
    assert 0 <= result.violation_fraction <= 1
    assert 0 <= result.weighted_volume <= 0.5
    assert result.ci_low <= result.violation_fraction <= result.ci_high

def test_perturbation_respects_ranges():
    p = ExperimentPoint(-0.75, 0.3, 1.0, np.pi/2, np.pi, np.pi/2, "original")
    rng = np.random.default_rng(1)
    q = perturb_point(p, rng, radius=0.2)
    assert -1 <= q.t_a <= 1
    assert -1 <= q.t_b <= 1
    assert 0 <= q.R <= 1
    assert 0 <= q.theta <= np.pi
    assert 0 <= q.phi <= 2*np.pi
    assert 0 <= q.gamma <= np.pi/2

def test_critical_threshold():
    values = np.array([0.0, 0.1, 0.2])
    fraction = np.array([0.0, 0.002, 0.02])
    assert critical_threshold(values, fraction, 0.005) == 0.2
