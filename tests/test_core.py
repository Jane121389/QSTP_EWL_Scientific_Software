
import numpy as np
from qstp_ewl.core import entangler, strategy, final_state, ExperimentPoint, evaluate

def test_unitarity():
    for name in ["original","cnot","dcnot","bgate"]:
        J=entangler(name,np.pi/3)
        assert np.linalg.norm(J.conj().T@J-np.eye(4))<1e-10

def test_state_normalization():
    for name in ["original","cnot","dcnot","bgate"]:
        psi=final_state(-.4,.3,entangler(name,np.pi/4))
        assert abs(np.linalg.norm(psi)-1)<1e-10

def test_probability_range():
    p=evaluate(ExperimentPoint(-.75,.3,1,np.pi/2,np.pi,np.pi/2,"original"))
    assert all(0<=x<=1 for x in p if np.isfinite(x))
