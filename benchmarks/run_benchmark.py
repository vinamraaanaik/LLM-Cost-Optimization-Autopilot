"""Send the same prompts to every available model and record cost, latency, and output.

    python -m benchmarks.run_benchmark [--max-output-tokens 256] [--delay-ms 150]

Models that can't run (not pulled, hosted calls not opted in, missing key) are recorded as
skipped rather than failed. Results are written to benchmark_results/ (git-ignored).
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.core.catalog import CREDENTIAL_ENV, MODEL_REGISTRY, ModelConfig, paid_providers_enabled
from app.core.config import PROJECT_ROOT
from app.providers.unified import send_request

PROMPTS_FILE = Path(__file__).with_name("prompts.json")
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"
SUMMARY_FIELDS = (
    "model_id",
    "provider_name",
    "success_count",
    "failure_count",
    "skipped_count",
    "total_cost_usd",
    "average_latency_ms",
)


async def _ollama_skip_reason(model: ModelConfig, client: httpx.AsyncClient) -> str | None:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        response = await client.get(f"{host}/api/tags")
        response.raise_for_status()
    except httpx.HTTPError:
        return f"Ollama is not reachable at {host}."
    installed = {m.get("name") or m.get("model") for m in response.json().get("models", [])}
    if model.model_id not in installed:
        return f"Model {model.model_id!r} is not pulled. Run: ollama pull {model.model_id}"
    return None


async def skip_reason(model: ModelConfig, client: httpx.AsyncClient) -> str | None:
    """Why this model can't be benchmarked here, or None when it can."""
    if model.provider_name in CREDENTIAL_ENV:
        if not paid_providers_enabled():
            return "Hosted calls are disabled. Set LLM_ALLOW_PAID_PROVIDERS=true to opt in."
        key_name = CREDENTIAL_ENV[model.provider_name]
        return None if os.getenv(key_name) else f"{key_name} is not set in this environment."
    if model.provider_name == "ollama":
        return await _ollama_skip_reason(model, client)
    return None


async def benchmark_model(
    model: ModelConfig,
    prompts: list[dict],
    client: httpx.AsyncClient,
    *,
    max_output_tokens: int,
    delay_ms: int,
) -> list[dict]:
    records = []
    for index, prompt in enumerate(prompts):
        base = {
            "model_id": model.model_id,
            "provider_name": model.provider_name,
            "prompt_id": prompt["id"],
            "prompt": prompt["prompt"],
        }
        try:
            result = await send_request(
                prompt["prompt"], model, max_output_tokens=max_output_tokens, client=client
            )
        except Exception as error:  # record each provider failure and keep going
            message = f"{type(error).__name__}: {error}"
            records.append({**base, "status": "failed", "error": message})
            print(f"FAIL {model.model_id} {prompt['id']}: {message}")
        else:
            records.append(
                {
                    **base,
                    "status": "success",
                    "output_text": result.output_text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                    "latency_ms": result.latency_ms,
                    "cost_usd": result.cost_usd,
                }
            )
            print(
                f"OK {model.model_id} {prompt['id']}: {result.total_tokens} tokens, "
                f"${result.cost_usd:.8f}, {result.latency_ms:.1f} ms"
            )
        if delay_ms > 0 and index < len(prompts) - 1:
            await asyncio.sleep(delay_ms / 1_000)
    return records


def summarize(records: list[dict]) -> list[dict]:
    """One summary row per model."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["model_id"]].append(record)
    rows = []
    for model_id, model_records in grouped.items():
        successful = [r for r in model_records if r["status"] == "success"]
        latencies = [r["latency_ms"] for r in successful]
        average_latency = round(sum(latencies) / len(latencies), 3) if latencies else ""
        rows.append(
            {
                "model_id": model_id,
                "provider_name": model_records[0]["provider_name"],
                "success_count": len(successful),
                "failure_count": sum(r["status"] == "failed" for r in model_records),
                "skipped_count": sum(r["status"] == "skipped" for r in model_records),
                "total_cost_usd": round(sum(r["cost_usd"] for r in successful), 10),
                "average_latency_ms": average_latency,
            }
        )
    return rows


async def run(max_output_tokens: int, delay_ms: int) -> int:
    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    started_at = datetime.now(UTC)
    records: list[dict] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        for model in MODEL_REGISTRY.values():
            reason = await skip_reason(model, client)
            if reason:
                print(f"SKIP {model.model_id}: {reason}")
                records += [
                    {
                        "model_id": model.model_id,
                        "provider_name": model.provider_name,
                        "prompt_id": prompt["id"],
                        "status": "skipped",
                        "error": reason,
                    }
                    for prompt in prompts
                ]
                continue
            records += await benchmark_model(
                model, prompts, client, max_output_tokens=max_output_tokens, delay_ms=delay_ms
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    results_path = RESULTS_DIR / f"run_{stamp}.json"
    summary_path = RESULTS_DIR / f"summary_{stamp}.csv"
    results_path.write_text(
        json.dumps(
            {
                "started_at": started_at.isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
                "prompt_count": len(prompts),
                "model_count": len(MODEL_REGISTRY),
                "model_configs": [
                    {
                        "provider_name": m.provider_name,
                        "model_id": m.model_id,
                        "cost_per_input_token_usd": m.cost_per_input_token,
                        "cost_per_output_token_usd": m.cost_per_output_token,
                        "quality_tier": m.quality_tier.value,
                    }
                    for m in MODEL_REGISTRY.values()
                ],
                "results": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summarize(records))

    expected = len(MODEL_REGISTRY) * len(prompts)
    succeeded = sum(record["status"] == "success" for record in records)
    print(f"Completed {succeeded}/{expected} calls.")
    print(f"Detailed results: {results_path}")
    print(f"Summary: {summary_path}")
    if succeeded == 0:
        print("No model completed the benchmark; inspect skipped and failed rows.")
        return 1
    if succeeded != expected:
        print("Partial baseline saved; inspect skipped and failed rows before comparing models.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Send the same prompts to each model.")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--delay-ms", type=int, default=150)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.max_output_tokens, args.delay_ms)))


if __name__ == "__main__":
    main()
