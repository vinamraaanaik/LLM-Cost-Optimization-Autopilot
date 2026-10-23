# Load Test and Cost Scenario Report

**Run ID:** `20261020T185430663907Z`  
**Run date:** 2026-10-20 18:54 UTC  
**Evidence class:** simulated integration/load run; not live model inference

> The test exercised the real FastAPI `/v1/completions` endpoint, classifier, routing logic, and isolated SQLite database with 1,000 unique prompts. A deterministic local stub replaced Ollama, so no model server was needed. The stub added synthetic 60–160 ms delays and estimated tokens as ceil(characters / 4). No hosted provider was enabled and no API keys were used.

## Load results

| Metric | Result |
|---|---:|
| Requests | 1,000 |
| Successful / failed | 1,000 / 0 |
| Unique prompts | 1,000 |
| Task families | 18 |
| Concurrency | 16 |
| Run duration | 11.92 s |
| Throughput | 83.92 successful requests/s (includes simulated delay) |
| End-to-end latency p50 / p95 / p99 | 170.75 ms/231.12 ms/265.75 ms |
| Synthetic provider latency p50 / p95 | 159.13 ms/218.75 ms |

### Routing and prompt mix

| Complexity | Intended prompt count | Classifier result count |
|---|---:|---:|
| Simple | 336 | 336 |
| Moderate | 334 | 363 |
| Complex | 330 | 301 |

Intended-tier agreement on this generated workload: **95.70%** (scenario labels, not a human-reviewed quality benchmark).

| Routed model | Requests |
|---|---:|
| `llama3.1:8b` | 664 |
| `llama3.2:1b` | 336 |

Task-family counts and per-request routing details are in `run-metadata.json` and `request-ledger.csv`.

## Cost savings scenario

| Metric | Result |
|---|---:|
| Synthetic input / output tokens | 45,901 / 22,000 |
| Configured local API charge | $0.00000000 |
| Hypothetical GPT-4o spend at configured rates | $0.33475250 |
| Hypothetical API-fee savings | $0.33475250 |
| Hypothetical API-fee reduction | 100.00% |
| GPT-4o configured rates | $2.5/M input, $10.0/M output |
| Pricing configuration checked | 2026-10-20 |

**Interpretation:** this 100% figure is only a pricing-model scenario: the selected local models have zero provider API fees in the checked-in configuration. The run used synthetic token counts, not a tokenizer or measured Ollama usage. It excludes hardware amortization, electricity, and any quality difference. It is not an observed cost reduction and should not be presented as a real savings result.

## What this run establishes

- The API accepted and logged the generated request volume through the normal classification and routing path.
- The SQLite audit trail recorded token and cost fields in an isolated database.
- `/v1/stats` computed its GPT-4o comparison from the recorded token counts without making model calls.
- Provider quality was not measured; response bodies were synthetic. Verification was disabled for this run.
- Throughput and latency are application-path measurements with a controlled stub delay, not Ollama performance figures.

## Artifacts

A run writes these files next to its report under `benchmark_results/` (git-ignored):

- `request-ledger.csv`: one privacy-safe row per generated prompt, including its SHA-256 hash.
- `run-metadata.json`: run settings, distributions, latency percentiles, cost scenario, and API stats.
- `load-test.sqlite3`: isolated request database used for the run.
- Re-run with: `python -m benchmarks.run_load_test --requests 1000 --concurrency 16`.
