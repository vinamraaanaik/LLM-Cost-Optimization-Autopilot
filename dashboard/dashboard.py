"""Local Streamlit dashboard for LLM Cost Autopilot telemetry.

    streamlit run dashboard/dashboard.py

Metrics are computed locally from SQLite and the configured prices. The API is only contacted
for actions: generating, refreshing verification, leaving feedback, and running evaluations.
"""

import os

import httpx
import pandas as pd
import streamlit as st

from app.core.catalog import MODEL_REGISTRY
from app.core.costing import savings_vs_baseline
from app.core.database import PENDING_VERIFICATION, Database
from app.core.drift import detect_drift
from app.core.quality import REQUEST_TYPES
from app.core.sample_data import sample_requests

API_URL = os.getenv("AUTOPILOT_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_BASELINE_ID = "gpt-4o"
SCORE_BINS = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
SCORE_LABELS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
COST_SERIES = ["Routed spend", "Hypothetical baseline"]
REQUEST_COLUMNS = [
    "created_at",
    "prompt_hash",
    "original_model_id",
    "model_id",
    "complexity",
    "verification_status",
    "quality_score",
    "escalated",
    "escalation_cost_delta_usd",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_ms",
    "feedback_rating",
    "prompt",
]
FAILURE_COLUMNS = [
    "created_at",
    "request_type",
    "complexity",
    "target_complexity",
    "original_model_id",
    "verifier_model_id",
    "escalation_model_id",
    "quality_score",
    "quality_threshold",
    "quality_gap",
    "escalation_cost_delta_usd",
    "reason",
    "prompt",
]


@st.cache_resource
def get_database() -> Database:
    """One shared handle; Streamlit re-runs this script on every interaction."""
    return Database()


def api_request(method: str, path: str, **kwargs):
    with httpx.Client(timeout=kwargs.pop("timeout", 900.0)) as client:
        response = client.request(method, f"{API_URL}{path}", **kwargs)
        response.raise_for_status()
        return response.json()


def show_table(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(frame, hide_index=True, use_container_width=True, **kwargs)


def score_histogram(scores: pd.Series, name: str) -> pd.Series:
    buckets = pd.cut(scores.dropna(), bins=SCORE_BINS, labels=SCORE_LABELS)
    return buckets.value_counts(sort=False).rename(name)


def load_request_frame(database: Database) -> tuple[pd.DataFrame, bool]:
    """Stored requests as a typed frame, or in-memory sample rows when the database is empty."""
    stored = database.cost_dashboard_requests()
    frame = pd.DataFrame(stored or sample_requests())
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    for column in ("input_tokens", "output_tokens", "cost_usd", "escalated", "latency_ms"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["quality_score"] = pd.to_numeric(frame["quality_score"], errors="coerce")
    frame["day"] = frame["created_at"].dt.tz_convert(None).dt.normalize()
    frame["week"] = frame["day"].dt.to_period("W-SUN").dt.start_time
    return frame, not stored


def cost_rollup(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    """Spend, baseline spend, and escalation counts per day or week."""
    return frame.groupby(period).agg(
        **{
            "Routed spend": ("cost_usd", "sum"),
            "Hypothetical baseline": ("baseline_cost_usd", "sum"),
        },
        request_count=("id", "count"),
        escalation_count=("escalated", "sum"),
    )


def render_alerts_and_summary(database: Database) -> None:
    drift = detect_drift(database.evaluation_scores())
    if drift["status"] == "degradation_detected":
        st.error(
            f"Evaluation drift detected: recent score {drift['recent_score']:.2f}, "
            f"baseline {drift['baseline_score']:.2f}."
        )
    elif drift["status"] == "insufficient_data":
        st.info(
            f"Drift monitoring needs {drift['samples_required']} evaluation results; "
            f"{drift['samples_available']} recorded so far."
        )

    metrics = database.metrics()
    st.caption("Stored telemetry summary from SQLite")
    cards = st.columns(8)
    cards[0].metric("Stored requests", metrics["request_count"])
    cards[1].metric("Stored spend", f"${metrics['total_cost_usd']:.6f}")
    cards[2].metric("Avg latency", f"{metrics['average_latency_ms']:.0f} ms")
    cards[3].metric("Eval score", f"{metrics['average_evaluation_score']:.0%}")
    cards[4].metric("Routing failures", metrics["routing_failure_count"])
    cards[5].metric("Escalations", metrics["escalation_count"])
    cards[6].metric("Verification pending", metrics["verification_pending_count"])
    feedback = f"{metrics['average_feedback_rating']:.1f}/5" if metrics["feedback_count"] else "—"
    cards[7].metric("Feedback", feedback)


def choose_baseline() -> tuple[str, float, float]:
    """Pick the all-traffic baseline model and (editable) per-million-token rates."""
    model_ids = list(MODEL_REGISTRY)
    default = DEFAULT_BASELINE_ID if DEFAULT_BASELINE_ID in model_ids else model_ids[0]
    baseline_id = st.selectbox(
        "Hypothetical all-traffic baseline model",
        options=model_ids,
        index=model_ids.index(default),
        help="A local estimate using the configured price per million tokens. No model is called.",
    )
    baseline = MODEL_REGISTRY[baseline_id]
    with st.expander("Configure baseline pricing", expanded=False):
        input_column, output_column = st.columns(2)
        input_rate = input_column.number_input(
            "Baseline input USD / 1M tokens",
            min_value=0.0,
            value=round(baseline.cost_per_input_token * 1_000_000, 6),
            step=0.1,
            format="%.4f",
            key=f"baseline_input_{baseline_id}",
        )
        output_rate = output_column.number_input(
            "Baseline output USD / 1M tokens",
            min_value=0.0,
            value=round(baseline.cost_per_output_token * 1_000_000, 6),
            step=0.1,
            format="%.4f",
            key=f"baseline_output_{baseline_id}",
        )
    return baseline_id, input_rate, output_rate


def render_cost_overview(database: Database) -> None:
    st.subheader("Cost and routing overview")
    frame, using_sample_data = load_request_frame(database)
    if using_sample_data:
        st.warning(
            "Sample data is shown because this database has no recorded inference requests. "
            "These illustrative rows exist only in memory and are not saved to SQLite."
        )

    baseline_id, input_rate, output_rate = choose_baseline()
    frame["baseline_cost_usd"] = (
        frame["input_tokens"] * input_rate + frame["output_tokens"] * output_rate
    ) / 1_000_000
    actual_total = float(frame["cost_usd"].sum())
    baseline_total = float(frame["baseline_cost_usd"].sum())
    savings, reduction_pct = savings_vs_baseline(actual_total, baseline_total)
    st.caption(
        "Actual spend is the locally recorded request cost, including any verification calls. "
        "The hypothetical baseline reprices the same recorded token counts at the selected "
        "rates. All values are estimates; local hardware and electricity are excluded."
    )

    saving = savings >= 0
    headline = st.columns(4)
    headline[0].metric(
        f"Estimated cost {'reduction' if saving else 'increase'} vs {baseline_id}",
        f"{abs(reduction_pct or 0.0):.1f}%",
        delta=(
            f"${abs(savings):.6f} estimated savings"
            if saving
            else f"${abs(savings):.6f} above baseline"
        ),
    )
    headline[1].metric("Routed spend", f"${actual_total:.6f}")
    headline[2].metric(f"Hypothetical {baseline_id} spend", f"${baseline_total:.6f}")
    headline[3].metric("Recorded requests", f"{len(frame):,}")

    daily, weekly = cost_rollup(frame, "day"), cost_rollup(frame, "week")
    daily_column, weekly_column = st.columns(2)
    with daily_column:
        st.markdown("**Estimated cost per day**")
        st.line_chart(daily[COST_SERIES])
    with weekly_column:
        st.markdown("**Estimated cost per week**")
        st.bar_chart(weekly[COST_SERIES])

    routing_column, quality_column, escalation_column = st.columns(3)
    with routing_column:
        st.markdown("**Initial routing distribution**")
        st.bar_chart(frame["original_model_id"].value_counts().rename("Requests"))
    with quality_column:
        st.markdown("**Verifier quality score distribution**")
        if frame["quality_score"].dropna().empty:
            st.caption("No verifier quality scores have been recorded yet.")
        else:
            st.bar_chart(score_histogram(frame["quality_score"], "Requests"))
    with escalation_column:
        st.markdown("**Escalation rate over time**")
        escalation_rate = daily["escalation_count"] / daily["request_count"] * 100
        st.line_chart(escalation_rate.rename("Escalation rate (%)"))

    st.markdown("**Golden-set evaluation score distribution**")
    evaluations = pd.DataFrame(database.dashboard_evaluations())
    if evaluations.empty:
        st.caption("No golden-set evaluation scores are stored in this database yet.")
    else:
        scores = pd.to_numeric(evaluations["score"], errors="coerce")
        st.bar_chart(score_histogram(scores, "Evaluation cases"))


def render_router_playground() -> None:
    st.subheader("Try the router")
    prompt = st.text_area("Prompt", height=120, placeholder="Ask a question or describe a task…")
    request_type = st.selectbox(
        "Request type",
        REQUEST_TYPES,
        help="Auto detects common task types; explicit types select a matching quality threshold.",
    )
    verification_budget_ms = st.slider(
        "Wait for verification before returning (ms)",
        min_value=0,
        max_value=10_000,
        value=0,
        step=250,
        help=(
            "Zero returns the first answer immediately and verifies in the background. "
            "A positive budget can return an escalated answer if verification finishes in time."
        ),
    )
    if st.button("Route and generate", type="primary", disabled=not prompt.strip()):
        try:
            st.session_state["last_result"] = api_request(
                "POST",
                "/v1/generate",
                json={
                    "prompt": prompt,
                    "max_output_tokens": 512,
                    "request_type": request_type,
                    "verification_budget_ms": verification_budget_ms,
                },
            )
        except httpx.HTTPStatusError as error:
            st.error(error.response.text)
        except httpx.HTTPError as error:
            st.error(f"Request failed: {error}")

    last = st.session_state.get("last_result")
    if last:
        render_generation_result(last)


def render_generation_result(last: dict) -> None:
    st.markdown(
        f"**{last['model_id']}** · {last['complexity']} complexity · "
        f"${last['cost_usd']:.8f} · {last['latency_ms']:.0f} ms"
    )
    st.caption(last["route_reason"])
    score = last["quality_score"]
    agreement = f" · agreement {score:.0%}" if score is not None else ""
    st.caption(f"Quality check: {last['verification_status']} · {last['request_type']}{agreement}")
    if last["escalated"]:
        st.info(
            f"Escalated from {last['original_model_id']} to {last['escalation_model_id']} "
            f"after a {last['quality_gap']:.0%} quality gap; added cost delta "
            f"${last['escalation_cost_delta_usd']:.8f}."
        )
    st.write(last["output_text"])

    if last["verification_status"] in PENDING_VERIFICATION and st.button(
        "Refresh verification result"
    ):
        try:
            last.update(api_request("GET", f"/v1/requests/{last['request_id']}"))
            last["total_tokens"] = last["input_tokens"] + last["output_tokens"]
            st.session_state["last_result"] = last
            st.rerun()
        except httpx.HTTPError as error:
            st.error(f"Could not refresh verification: {error}")

    # Human feedback is stored with the request and shown in the history table below.
    with st.form("feedback_form"):
        rating = st.slider("How useful was this response?", min_value=1, max_value=5, value=4)
        comment = st.text_input("Optional note")
        if st.form_submit_button("Save feedback"):
            try:
                api_request(
                    "POST",
                    "/v1/feedback",
                    json={"request_id": last["request_id"], "rating": rating, "comment": comment},
                )
                st.success("Feedback saved.")
            except httpx.HTTPError as error:
                st.error(f"Could not save feedback: {error}")


def render_evaluation() -> None:
    st.subheader("Golden-set evaluation")
    st.caption("Runs deterministic checks; optionally add a local Ollama model as a judge.")
    local_models = [model_id for model_id, model in MODEL_REGISTRY.items() if model.local]
    judge = st.selectbox(
        "Optional local judge model",
        options=["Disabled", *local_models],
        help="Adds one local judge call per golden case and stores its score separately.",
    )
    if st.button("Run evaluation"):
        with st.spinner("Running golden prompts…"):
            try:
                st.session_state["last_evaluation"] = api_request(
                    "POST",
                    "/v1/evaluate",
                    json={"judge_model_id": None if judge == "Disabled" else judge},
                )
            except httpx.HTTPStatusError as error:
                st.error(error.response.text)
            except httpx.HTTPError as error:
                st.error(f"Evaluation failed: {error}")

    evaluation = st.session_state.get("last_evaluation")
    if evaluation:
        st.write(
            f"Run `{evaluation['run_id']}` · {evaluation['passed_count']}/"
            f"{evaluation['case_count']} cases passed · score {evaluation['average_score']:.0%}"
        )
        if evaluation["average_judge_score"] is not None:
            st.caption(f"Local judge score: {evaluation['average_judge_score']:.0%}")
        output_column = st.column_config.TextColumn("Model output", width="large")
        show_table(pd.DataFrame(evaluation["cases"]), column_config={"output_text": output_column})


def render_history(database: Database) -> None:
    st.subheader("Recent requests")
    requests = database.recent_requests(limit=100)
    if requests:
        show_table(pd.DataFrame(requests)[REQUEST_COLUMNS])
    else:
        st.caption("No stored requests yet.")

    st.subheader("Routing failure log")
    failures = database.recent_routing_failures(limit=100)
    if failures:
        show_table(pd.DataFrame(failures)[FAILURE_COLUMNS])
    else:
        st.caption("No routing failures have been detected.")


def main() -> None:
    st.set_page_config(page_title="LLM Cost Autopilot", page_icon="🧭", layout="wide")
    st.title("LLM Cost Autopilot")
    st.caption("Local-first model routing, cost tracking, and quality checks")
    st.info(
        "Dashboard metrics are calculated locally from SQLite and configured prices. The API is "
        "contacted only for generation, verification refresh, feedback, or evaluation actions."
    )
    database = get_database()
    render_alerts_and_summary(database)
    render_cost_overview(database)
    render_router_playground()
    render_evaluation()
    render_history(database)


if __name__ == "__main__":  # Streamlit executes the script as __main__
    main()
