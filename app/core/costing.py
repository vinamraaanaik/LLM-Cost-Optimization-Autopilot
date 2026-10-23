"""Token-cost arithmetic shared by routing, accounting, and reporting."""

from app.core.catalog import ModelConfig


def estimate_cost_usd(
    model: ModelConfig,
    input_tokens: int,
    output_tokens: int,
    *,
    input_weight: float = 1.0,
    output_weight: float = 1.0,
) -> float:
    """Estimated USD cost of one call.

    The weights are a routing heuristic (see ``configs/routing.yaml``); leave them at 1.0
    for real accounting.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")
    return round(
        input_tokens * model.cost_per_input_token * input_weight
        + output_tokens * model.cost_per_output_token * output_weight,
        10,
    )


def savings_vs_baseline(actual_usd: float, baseline_usd: float) -> tuple[float, float | None]:
    """Return (savings in USD, savings as a % of baseline); the % is None for a zero baseline."""
    savings = baseline_usd - actual_usd
    return savings, (savings / baseline_usd * 100 if baseline_usd > 0 else None)
