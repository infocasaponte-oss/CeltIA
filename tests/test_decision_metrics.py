import math
from celtia.decision.metrics import brier_score, expected_calibration_error, negative_log_likelihood

def test_brier_perfect(): assert brier_score([0.0,1.0],[0,1]) == 0.0
def test_nll_prefers_confident_correct(): assert negative_log_likelihood([0.1,0.9],[0,1]) < negative_log_likelihood([0.4,0.6],[0,1])
def test_ece_perfect_confidence(): assert math.isclose(expected_calibration_error([1.0,1.0],[True,True]),0.0)
