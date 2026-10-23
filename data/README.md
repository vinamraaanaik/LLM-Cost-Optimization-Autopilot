# Data files

Small, versioned data used by the application:

| File | Purpose |
|---|---|
| `models.json`, `pricing.json` | Model IDs, quality tiers, and per-million-token USD rates. Both must list the same models. |
| `quality_thresholds.json` | Minimum agreement score per request type for verification. |
| `golden_dataset.json` | Prompts with deterministic checks for repeatable evaluation. |
| `complexity_examples.tsv` | Hand-labeled prompts (`label`, `prompt`) that train the complexity classifier. |
| `complexity_model_metrics.json` | Hold-out report from `python -m scripts.train_complexity_classifier`. |

The classifier's features are computed from the prompt text at training and serving time, so
there is no derived training file to keep in sync. Routing failures recorded in the database
are added to the training data automatically.

`autopilot.sqlite3` is created at runtime and is git-ignored. It holds prompts, responses,
usage estimates, evaluations, feedback, verification jobs, and routing failures. Keep model
weights, credentials, and private production data out of this directory.

The golden set and classifier data are illustrative portfolio fixtures, not production
validation.
