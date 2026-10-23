"""Load-test the real FastAPI app with simulated inference.

    python -m benchmarks.run_load_test --requests 1000 --concurrency 16

Only Ollama's network call is replaced by a deterministic stub. The HTTP endpoint, classifier,
router, cost accounting, and SQLite path are all real. Token counts and latency are synthetic,
so the results must not be reported as live inference.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.core.config import CONFIG_DIR, DATA_DIR, PROJECT_ROOT
from app.providers.base import ProviderResult

# Vocabulary and prompt templates live in JSON so this file stays logic-only. Templates are
# (task family, intended tier, text) and are cycled in order, so a run is balanced across
# families. `{items}` is the only field that draws extra random values.
WORKLOAD = json.loads(Path(__file__).with_name("load_test_workload.json").read_text("utf-8"))

LEDGER_FIELDS = (
    "prompt_id",
    "prompt_hash",
    "task_family",
    "intended_complexity",
    "status_code",
    "complexity",
    "selected_model",
    "verification_status",
    "input_tokens_estimate",
    "output_tokens_estimate",
    "total_tokens_estimate",
    "provider_latency_ms_simulated",
    "end_to_end_ms",
    "routed_cost_usd_estimate",
    "error",
)


def build_prompt_set(count: int, seed: int) -> list[dict[str, str]]:
    """Unique prompts cycling through every template, so intended tiers stay balanced."""
    rng = random.Random(seed)
    templates = WORKLOAD["templates"]
    prompts = []
    for index in range(count):
        # The draw order is part of the seed's contract: changing it changes every prompt.
        fields = {
            "case": index + 1,
            "domain": rng.choice(WORKLOAD["domains"]),
            "person": rng.choice(WORKLOAD["people"]),
            "place": rng.choice(WORKLOAD["places"]),
            "item": rng.choice(WORKLOAD["items"]),
            "date": rng.choice(WORKLOAD["dates"]),
            "amount": rng.choice(WORKLOAD["amounts"]),
        }
        entry = templates[index % len(templates)]
        template = entry["template"]
        if "{items}" in template:
            extra = [rng.choice(WORKLOAD["items"]), rng.choice(WORKLOAD["items"])]
            fields["items"] = ", ".join([fields["item"], *extra])
        prompts.append(
            {
                "id": f"case-{index + 1:04d}",
                "family": entry["family"],
                "intended_complexity": entry["tier"],
                "prompt": template.format(**fields, amount_plus=fields["amount"] + 27),
            }
        )
    return prompts


class SimulatedOllamaAdapter:
    """Local-only stub with bounded synthetic latency and char-based token estimates."""

    async def complete(self, prompt, model_config, max_output_tokens, client):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        await asyncio.sleep((60 + int(digest[:4], 16) % 101) / 1_000)
        case_match = re.search(r"Case (\d{4})", prompt)
        case_id = case_match.group(1) if case_match else "unknown"
        text = (
            f"Simulated local response for case {case_id}. "
            f"The {model_config.model_id} adapter accepted the request."
        )
        return ProviderResult(
            output_text=text[: max(1, max_output_tokens * 4)],
            input_tokens=max(1, math.ceil(len(prompt) / 4)),
            output_tokens=max(1, math.ceil(len(text) / 4)),
        )


# ---- request phase ---------------------------------------------------------------------


async def submit(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, item: dict) -> dict:
    """Send one prompt through POST /v1/completions and return its ledger row."""
    row = {
        "prompt_id": item["id"],
        "prompt_hash": hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest(),
        "task_family": item["family"],
        "intended_complexity": item["intended_complexity"],
    }
    body = {
        "model": "gpt-4o",  # ignored by design; the router selects a local model
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": 128,
        "verify_quality": False,
    }
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post("/v1/completions", json=body)
            row["end_to_end_ms"] = round((time.perf_counter() - started) * 1_000, 2)
            payload = response.json()
            metadata, usage = payload.get("metadata", {}), payload.get("usage", {})
            row |= {
                "status_code": response.status_code,
                "complexity": metadata.get("complexity"),
                "selected_model": metadata.get("selected_model"),
                "verification_status": metadata.get("verification_status"),
                "input_tokens_estimate": usage.get("prompt_tokens"),
                "output_tokens_estimate": usage.get("completion_tokens"),
                "total_tokens_estimate": usage.get("total_tokens"),
                "provider_latency_ms_simulated": metadata.get("latency_ms"),
                "routed_cost_usd_estimate": metadata.get("cost_usd_estimate"),
            }
        except Exception as error:  # a failed request is a data point, not a crash
            row["end_to_end_ms"] = round((time.perf_counter() - started) * 1_000, 2)
            row |= {"status_code": "error", "error": f"{type(error).__name__}: {error}"}
    return row


# ---- summary and report (pure functions of the collected data) ----------------------------


@dataclass(frozen=True)
class RunSettings:
    run_id: str
    started_at: datetime
    completed_at: datetime
    duration_s: float
    concurrency: int
    seed: int


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)], 2)


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def summarize(
    prompts: list[dict],
    records: list[dict],
    *,
    settings: RunSettings,
    stats: dict,
    health: dict,
    models: list[dict],
    pricing: dict,
) -> dict:
    success = [row for row in records if row["status_code"] == 200]
    failures = [row for row in records if row["status_code"] != 200]
    complexity_counts = Counter(row.get("complexity") for row in success)
    tier_matches = sum(row.get("complexity") == row["intended_complexity"] for row in success)
    end_to_end = [float(row["end_to_end_ms"]) for row in records if "end_to_end_ms" in row]
    provider = [float(row["provider_latency_ms_simulated"]) for row in success]
    input_tokens = int(stats.get("input_tokens", 0))
    output_tokens = int(stats.get("output_tokens", 0))
    return {
        "run_id": settings.run_id,
        "started_at_utc": settings.started_at.isoformat(),
        "completed_at_utc": settings.completed_at.isoformat(),
        "method": (
            "in-process ASGI requests to the real FastAPI app; "
            "Ollama adapter replaced by a deterministic simulator"
        ),
        "inference_mode": "simulated; no actual model inference",
        "provider_latency_mode": "synthetic 60-160 ms delay",
        "token_count_mode": "synthetic ceil(characters / 4) estimates; not tokenizer usage",
        "request_count": len(records),
        "success_count": len(success),
        "failure_count": len(failures),
        "unique_prompt_count": len({row["prompt_hash"] for row in records}),
        "task_family_count": len({item["family"] for item in prompts}),
        "task_family_distribution": dict(sorted(Counter(i["family"] for i in prompts).items())),
        "intended_complexity_distribution": dict(
            sorted(Counter(i["intended_complexity"] for i in prompts).items())
        ),
        "classifier_complexity_distribution": dict(sorted(complexity_counts.items())),
        "intended_tier_matches": tier_matches,
        "intended_tier_agreement_pct": (
            round(tier_matches / len(success) * 100, 2) if success else None
        ),
        "routed_model_distribution": dict(
            sorted(Counter(row.get("selected_model") for row in success).items())
        ),
        "concurrency": settings.concurrency,
        "seed": settings.seed,
        "run_duration_seconds": round(settings.duration_s, 3),
        "successful_requests_per_second": (
            round(len(success) / settings.duration_s, 2) if settings.duration_s else 0.0
        ),
        "end_to_end_latency_ms": {
            "p50": percentile(end_to_end, 50),
            "p95": percentile(end_to_end, 95),
            "p99": percentile(end_to_end, 99),
            "mean": mean(end_to_end),
        },
        "synthetic_provider_latency_ms": {
            "p50": percentile(provider, 50),
            "p95": percentile(provider, 95),
            "mean": mean(provider),
        },
        "token_usage_estimates": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
        "cost_scenario_estimate_usd": {
            "routed_local_api_cost": round(float(stats.get("actual_estimated_cost_usd") or 0), 10),
            "all_gpt4o_hypothetical_baseline": round(
                float(stats.get("hypothetical_baseline_cost_usd") or 0), 10
            ),
            "hypothetical_api_fee_savings": round(
                float(stats.get("estimated_savings_usd") or 0), 10
            ),
            "hypothetical_api_fee_reduction_pct": stats.get("estimated_cost_reduction_pct"),
            "gpt4o_rates_per_million_tokens": pricing["models"]["gpt-4o"],
            "pricing_last_verified": pricing.get("last_verified"),
        },
        "api_stats_response": stats,
        "health_response": health,
        "paid_providers_enabled": False,
        "verification_enabled_for_run": False,
        "quality_evaluated": False,
        "failure_samples": failures[:10],
        "model_availability": [
            {"model_id": m["model_id"], "available_for_routing": m["available_for_routing"]}
            for m in models
        ],
    }


def md_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Metric | Result |", "|---|---:|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def render_report(summary: dict) -> str:
    """Markdown report. Every number comes from ``summary`` (the run-metadata.json content)."""
    started = datetime.fromisoformat(summary["started_at_utc"])
    e2e, provider = summary["end_to_end_latency_ms"], summary["synthetic_provider_latency_ms"]
    cost = summary["cost_scenario_estimate_usd"]
    tokens = summary["token_usage_estimates"]
    rates = cost["gpt4o_rates_per_million_tokens"]
    reduction = cost["hypothetical_api_fee_reduction_pct"]
    intended = summary["intended_complexity_distribution"]
    classified = summary["classifier_complexity_distribution"]
    count = summary["request_count"]

    load_table = md_table(
        [
            ("Requests", f"{count:,}"),
            (
                "Successful / failed",
                f"{summary['success_count']:,} / {summary['failure_count']:,}",
            ),
            ("Unique prompts", f"{summary['unique_prompt_count']:,}"),
            ("Task families", str(summary["task_family_count"])),
            ("Concurrency", str(summary["concurrency"])),
            ("Run duration", f"{summary['run_duration_seconds']:.2f} s"),
            (
                "Throughput",
                f"{summary['successful_requests_per_second']:.2f} successful requests/s "
                "(includes simulated delay)",
            ),
            (
                "End-to-end latency p50 / p95 / p99",
                "/".join(f"{e2e[key]} ms" for key in ("p50", "p95", "p99")),
            ),
            (
                "Synthetic provider latency p50 / p95",
                "/".join(f"{provider[key]} ms" for key in ("p50", "p95")),
            ),
        ]
    )
    tier_rows = "\n".join(
        f"| {tier.title()} | {intended.get(tier, 0):,} | {classified.get(tier, 0):,} |"
        for tier in ("simple", "moderate", "complex")
    )
    model_rows = "\n".join(
        f"| `{model_id}` | {n:,} |"
        for model_id, n in sorted(summary["routed_model_distribution"].items())
    )
    cost_table = md_table(
        [
            (
                "Synthetic input / output tokens",
                f"{tokens['input']:,} / {tokens['output']:,}",
            ),
            ("Configured local API charge", f"${cost['routed_local_api_cost']:.8f}"),
            (
                "Hypothetical GPT-4o spend at configured rates",
                f"${cost['all_gpt4o_hypothetical_baseline']:.8f}",
            ),
            ("Hypothetical API-fee savings", f"${cost['hypothetical_api_fee_savings']:.8f}"),
            (
                "Hypothetical API-fee reduction",
                f"{reduction:.2f}%" if reduction is not None else "Unavailable",
            ),
            (
                "GPT-4o configured rates",
                f"${rates['input']}/M input, ${rates['output']}/M output",
            ),
            ("Pricing configuration checked", str(cost["pricing_last_verified"] or "unknown")),
        ]
    )
    lines = [
        "# Load Test and Cost Scenario Report",
        "",
        f"**Run ID:** `{summary['run_id']}`  ",
        f"**Run date:** {started.strftime('%Y-%m-%d %H:%M UTC')}  ",
        "**Evidence class:** simulated integration/load run; not live model inference",
        "",
        (
            "> The test exercised the real FastAPI `/v1/completions` endpoint, classifier, "
            f"routing logic, and isolated SQLite database with {count:,} unique prompts. "
            "A deterministic local stub replaced Ollama, so no model server was needed. "
            "The stub added synthetic 60–160 ms delays and estimated tokens as "
            "ceil(characters / 4). No hosted provider was enabled and no API keys were used."
        ),
        "",
        "## Load results",
        "",
        load_table,
        "",
        "### Routing and prompt mix",
        "",
        "| Complexity | Intended prompt count | Classifier result count |",
        "|---|---:|---:|",
        tier_rows,
        "",
        (
            "Intended-tier agreement on this generated workload: "
            f"**{summary['intended_tier_agreement_pct']:.2f}%** "
            "(scenario labels, not a human-reviewed quality benchmark)."
        ),
        "",
        "| Routed model | Requests |",
        "|---|---:|",
        model_rows,
        "",
        (
            "Task-family counts and per-request routing details are in "
            "`run-metadata.json` and `request-ledger.csv`."
        ),
        "",
        "## Cost savings scenario",
        "",
        cost_table,
        "",
        (
            "**Interpretation:** this 100% figure is only a pricing-model scenario: the "
            "selected local models have zero provider API fees in the checked-in "
            "configuration. The run used synthetic token counts, not a tokenizer or measured "
            "Ollama usage. It excludes hardware amortization, electricity, and any quality "
            "difference. It is not an observed cost reduction and should not be presented as "
            "a real savings result."
        ),
        "",
        "## What this run establishes",
        "",
        (
            "- The API accepted and logged the generated request volume through the normal "
            "classification and routing path."
        ),
        "- The SQLite audit trail recorded token and cost fields in an isolated database.",
        (
            "- `/v1/stats` computed its GPT-4o comparison from the recorded token counts "
            "without making model calls."
        ),
        (
            "- Provider quality was not measured; response bodies were synthetic. "
            "Verification was disabled for this run."
        ),
        (
            "- Throughput and latency are application-path measurements with a controlled "
            "stub delay, not Ollama performance figures."
        ),
        "",
        "## Artifacts",
        "",
        "A run writes these files next to its report under `benchmark_results/` (git-ignored):",
        "",
        (
            "- `request-ledger.csv`: one privacy-safe row per generated prompt, including "
            "its SHA-256 hash."
        ),
        (
            "- `run-metadata.json`: run settings, distributions, latency percentiles, cost "
            "scenario, and API stats."
        ),
        "- `load-test.sqlite3`: isolated request database used for the run.",
        (
            "- Re-run with: "
            "`python -m benchmarks.run_load_test --requests 1000 --concurrency 16`."
        ),
        "",
    ]
    return "\n".join(lines)


# ---- orchestration ------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> Path:
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_root / f"load-test-{run_id}"
    output_dir.mkdir(parents=True)
    print(f"[load-test] isolated output: {output_dir}", flush=True)

    # Isolate all database writes and keep hosted providers disabled. This must happen before
    # app.main is imported, because the app opens its database at import time.
    os.environ["LLM_AUTOPILOT_DB"] = str(output_dir / "load-test.sqlite3")
    os.environ["LLM_ROUTING_RULES_PATH"] = str(CONFIG_DIR / "routing.yaml")
    os.environ["LLM_ALLOW_PAID_PROVIDERS"] = "false"
    os.environ["LLM_VERIFICATION_MODE"] = "inline"
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)

    import app.main as main_module
    from app.providers import unified

    unified.PROVIDERS["ollama"] = SimulatedOllamaAdapter()
    prompts = build_prompt_set(args.requests, args.seed)
    semaphore = asyncio.Semaphore(args.concurrency)
    print(f"[load-test] app ready; submitting {len(prompts)} requests", flush=True)

    async with (
        main_module.lifespan(main_module.app),  # fits the classifier, as in production
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_module.app),
            base_url="http://autopilot.local",
            timeout=httpx.Timeout(30.0),
        ) as client,
    ):

        async def send_with_shared_client(prompt, model_config, max_output_tokens=512):
            return await unified.send_request(
                prompt, model_config, max_output_tokens=max_output_tokens, client=client
            )

        # One shared client avoids constructing a new TLS context for every call.
        main_module.send_request = send_with_shared_client
        health = (await client.get("/health")).json()

        wall_started = time.perf_counter()
        records = await asyncio.gather(*(submit(client, semaphore, item) for item in prompts))
        duration_s = time.perf_counter() - wall_started
        print(f"[load-test] request phase done in {duration_s:.2f}s; reading summary", flush=True)
        stats = (await client.get("/v1/stats")).json()
        models = (await client.get("/v1/models")).json()

    pricing = json.loads((DATA_DIR / "pricing.json").read_text(encoding="utf-8"))
    summary = summarize(
        prompts,
        records,
        settings=RunSettings(
            run_id, started_at, datetime.now(UTC), duration_s, args.concurrency, args.seed
        ),
        stats=stats,
        health=health,
        models=models,
        pricing=pricing,
    )

    with (output_dir / "request-ledger.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "run-metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "load-test-report.md"
    report_path.write_text(render_report(summary), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1_000, help="number of generated requests")
    parser.add_argument("--concurrency", type=int, default=16, help="maximum in-flight requests")
    parser.add_argument("--seed", type=int, default=20261021, help="prompt and delay seed")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results",
        help="directory for isolated run artifacts",
    )
    args = parser.parse_args()
    if not 1 <= args.requests <= 10_000:
        parser.error("--requests must be between 1 and 10000")
    if not 1 <= args.concurrency <= 128:
        parser.error("--concurrency must be between 1 and 128")
    return args


if __name__ == "__main__":
    print("[load-test] starting", flush=True)
    print(f"Load test complete: {asyncio.run(run(parse_args()))}")
