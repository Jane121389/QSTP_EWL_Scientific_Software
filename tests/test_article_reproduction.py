
import numpy as np
from qstp_ewl.core import ExperimentPoint, evaluate

def test_complementary_article_endpoints():
    expected = {
        19*np.pi/30: 0.940,
        17*np.pi/30: 0.792,
        np.pi/2: 0.619,
    }
    for theta, target in expected.items():
        p = evaluate(ExperimentPoint(-0.25, 0.30, 1.0, theta, 0.0, np.pi/2, "original"))
        assert abs(p.p_d - target) < 0.02
