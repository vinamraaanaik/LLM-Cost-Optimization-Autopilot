"""Moving-window drift alert for golden-set evaluation scores."""


def detect_drift(
    scores: list[float], *, window_size: int = 10, degradation_threshold: float = 0.15
) -> dict:
    """Compare the newest window of scores with the one before it.

    ``scores`` must be newest first (the order the database returns them). Needs two full
    windows before it will alert, so a handful of noisy runs cannot trigger it.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    required = window_size * 2
    if len(scores) < required:
        return {
            "is_drifting": False,
            "status": "insufficient_data",
            "baseline_score": None,
            "recent_score": None,
            "degradation": None,
            "samples_required": required,
            "samples_available": len(scores),
        }
    recent_mean = sum(scores[:window_size]) / window_size
    baseline_mean = sum(scores[window_size:required]) / window_size
    degradation = baseline_mean - recent_mean
    drifting = degradation >= degradation_threshold
    return {
        "is_drifting": drifting,
        "status": "degradation_detected" if drifting else "stable",
        "baseline_score": round(baseline_mean, 4),
        "recent_score": round(recent_mean, 4),
        "degradation": round(degradation, 4),
        "threshold": degradation_threshold,
        "samples_required": required,
        "samples_available": len(scores),
    }
