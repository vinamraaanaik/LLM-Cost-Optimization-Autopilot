"""Evaluate the complexity classifier and refresh its hold-out report.

Trains on the hand-labeled examples plus routing failures stored in the database
(LLM_AUTOPILOT_DB), then writes data/complexity_model_metrics.json.

    python -m scripts.train_complexity_classifier
"""

import json

from app.core.complexity import evaluate_classifier, training_samples
from app.core.config import DATA_DIR
from app.core.database import Database

METRICS_PATH = DATA_DIR / "complexity_model_metrics.json"


def main() -> None:
    metrics = evaluate_classifier(training_samples(Database().training_failures()))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"{metrics['dataset_size']} examples {metrics['source_counts']}; hold-out accuracy "
        f"{metrics['accuracy']:.1%} on {metrics['test_size']} prompts."
    )
    print("Confusion matrix (rows = actual, columns = predicted; simple, moderate, complex):")
    for row in metrics["confusion_matrix"]:
        print(" ".join(f"{count:3d}" for count in row))


if __name__ == "__main__":
    main()
