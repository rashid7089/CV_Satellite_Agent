"""Evaluate the saved model and write reproducible production metrics."""
import json
import time
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "processed" / "test"
MODEL_PATH = ROOT / "models" / "ResNet_history.keras"
LABELS_PATH = ROOT / "models" / "labels.json"
REPORT_PATH = ROOT / "reports" / "model_metrics.json"


def main() -> None:
    if not TEST_DIR.exists():
        raise SystemExit(f"Test split not found: {TEST_DIR}. Run training/dataset.py first.")
    labels_raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = [labels_raw[k] for k in sorted(labels_raw, key=int)]
    dataset = keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, image_size=(128, 128), batch_size=16
    ).map(lambda x, y: (x / 255.0, y))
    model = keras.models.load_model(MODEL_PATH, compile=False)
    started = time.perf_counter()
    probabilities = model.predict(dataset, verbose=1)
    elapsed = time.perf_counter() - started
    truth = np.concatenate([y.numpy() for _, y in dataset])
    predicted = np.argmax(probabilities, axis=1)
    report = classification_report(truth, predicted, target_names=labels, output_dict=True, zero_division=0)
    metrics = {
        "model_name": "ResNet50V2",
        "model_version": "1.0.0",
        "seed": SEED,
        "test_images": int(len(truth)),
        "test_accuracy": round(float((truth == predicted).mean()), 6),
        "precision_macro": round(float(report["macro avg"]["precision"]), 6),
        "recall_macro": round(float(report["macro avg"]["recall"]), 6),
        "f1_macro": round(float(report["macro avg"]["f1-score"]), 6),
        "evaluation_seconds": round(elapsed, 3),
        "average_inference_ms": round(elapsed * 1000 / len(truth), 3),
        "confusion_matrix": confusion_matrix(truth, predicted).tolist(),
        "per_class": {name: report[name] for name in labels},
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
