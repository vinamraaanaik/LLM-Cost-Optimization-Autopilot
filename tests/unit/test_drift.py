import pytest

from app.core.drift import detect_drift


def test_drift_detects_a_drop_in_recent_scores():
    # Newest first: the ten most recent scores are far below the ten before them.
    result = detect_drift([0.4] * 10 + [0.9] * 10, window_size=10)
    assert result["is_drifting"]
    assert result["status"] == "degradation_detected"
    assert result["degradation"] == 0.5


def test_stable_scores_do_not_alert():
    assert detect_drift([0.9] * 20)["status"] == "stable"


def test_improvement_does_not_alert():
    assert not detect_drift([0.9] * 10 + [0.4] * 10)["is_drifting"]


def test_drift_waits_for_enough_data():
    result = detect_drift([1.0] * 3, window_size=2)
    assert result["status"] == "insufficient_data"
    assert not result["is_drifting"]
    assert (result["samples_available"], result["samples_required"]) == (3, 4)


def test_window_size_must_be_positive():
    with pytest.raises(ValueError):
        detect_drift([1.0], window_size=0)
