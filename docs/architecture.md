# Architecture

The component diagram is in the [README](../README.md#how-it-works). This document covers what it leaves out: the request lifecycle, the data model, how concurrency is handled, the error contract, and where to extend the system.

## Request lifecycle

`POST /v1/generate` (and `/v1/completions`, which wraps it):

```mermaid
sequenceDiagram
  participant C as Client
  participant A as API
  participant R as Classifier + Router
  participant P as Provider
  participant D as SQLite
  participant V as Verifier (inline task or worker)
  C->>A: prompt
  A->>R: classify, then choose model
  R-->>A: model, reason, estimated cost
  A->>P: send_request
  P-->>A: text, tokens, latency, cost
  A->>D: record request (verification_status = pending)
  alt verify_quality
    A->>V: schedule (inline task or queue row)
    opt verification_budget_ms > 0
      Note over A,V: API waits up to the budget
    end
  end
  A->>D: re-read the row
  A-->>C: current answer (original, or escalated if already settled)
  V->>P: ask the next tier up
  V->>D: score; if below threshold, replace answer and log failure
```

Two details matter:

- **The response is built from the database row, not from the first call.** If verification finishes within the budget, the caller gets the escalated answer without any special casing.
- **The first answer is never lost.** `initial_output_text` keeps it for audit even after escalation replaces `output_text`.

## Verification and escalation

1. **Pick a verifier.** The cheapest available model in the highest tier above the one that answered. With hosted calls disabled, a local 1B answer is checked by the local 8B model; a local 8B answer has no verifier and is marked `skipped`.
2. **Score agreement.** Thresholds come from `data/quality_thresholds.json` by request type. Classification compares normalized labels, extraction compares each requested JSON field and value, and everything else uses word-set F1 averaged with sequence similarity.
3. **Escalate on failure.** The verifier's answer, already generated from the same prompt, becomes the final answer, so a failure costs one extra call rather than two. The request row records both models, the quality gap, the verifier cost and the cost delta.
4. **Record the failure.** A `routing_failures` row stores the prompt, the tier it should have gone to (`simple` to `moderate`, `moderate` to `complex`, `complex` stays), and the features at that moment.
5. **Retrain.** On startup and on a schedule the API refits the classifier from the labelled TSV plus these rows. The model is swapped in as one object, so in-flight requests never see a half-trained pipeline.

Every verification outcome ends in a terminal status (`verified`, `escalated`, `skipped`, `error`); a verifier failure never changes the answer the user already received.

## Concurrency and consistency

| Concern | Approach |
|---|---|
| Several processes share one SQLite file | Short-lived connections and WAL mode, so readers do not block the writer. |
| Two workers claim the same job | `BEGIN IMMEDIATE` takes the write lock before the `SELECT`, so claims are serialised. |
| A worker dies mid-job | Jobs are leased; an expired lease makes the job claimable again and increments `attempts`. |
| Duplicate enqueue | `ON CONFLICT DO NOTHING` on `request_id`. |
| Concurrent routing-rule updates | Validate, write a temp file in the same directory, then `os.replace`, so readers never see a partial file. |
| Background tasks being collected | `_VERIFICATION_TASKS` holds a reference until each task finishes. |

## Data model

| Table | Holds |
|---|---|
| `requests` | One row per call: prompt and its SHA-256, routed and final model, tokens, latency, cost, verification state, escalation details. |
| `feedback` | One human rating per request (upserted). |
| `evaluations` | Golden-set results per run, with optional judge score and cost. |
| `routing_failures` | Prompts the router got wrong, labelled with the tier they needed. |
| `verification_jobs` | The queue: status, attempts, lease expiry, last error. |

Databases from earlier versions are upgraded in place on startup (`_ADDED_COLUMNS` and a backfill in `app/core/database.py`).

## Error contract

| Situation | Status |
|---|---|
| Unknown `model_id` / `judge_model_id` / `request_id` | 404 |
| Invalid routing config, or a message list with no user turn | 422 |
| Streaming requested, or a hosted judge model | 400 |
| Hosted provider disabled or missing a key, or no eligible model | 503 |
| Any other provider failure | 502, naming only the exception type so internal hostnames never reach clients |

## Extending

- **New provider:** implement `ProviderAdapter.complete`, register it in `PROVIDERS`, then add the model to `data/models.json` and `data/pricing.json` (the two must list the same models).
- **New task type:** add it to the `RequestType` literal in `app/core/quality.py` and give it an entry in `data/quality_thresholds.json`; a test fails if the two disagree. Add a pattern to `_TYPE_PATTERNS` if it should be auto-detected from the prompt.
- **New golden check:** add a function to `_CHECKS` in `app/core/evaluation.py`.
- **New feature for the classifier:** add it to `extract_features` and `FEATURE_NAMES`, refresh `tests/unit/test_complexity.py`'s pinned vectors, and rerun `python -m scripts.train_complexity_classifier`. Because features are never stored, nothing else needs migrating.
