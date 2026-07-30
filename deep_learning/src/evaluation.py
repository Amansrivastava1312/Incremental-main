import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from deep_learning.src.training import ARTIFACTS, DATA

# ARTIFACTS = os.path.join(BASE, "artifacts")


class Evaluator:
    def __init__(self):
        self.classes = sorted(os.listdir(os.path.join(DATA, "train")))

    def _metrics(self, model, test_ds):
        y_true = np.concatenate([y.numpy() for _, y in test_ds])
        y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(
                float(
                    precision_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "recall": round(
                float(
                    recall_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "f1": round(
                float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "report": classification_report(
                y_true,
                y_pred,
                target_names=self.classes,
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def compare(self, test_ds):
        os.makedirs(ARTIFACTS, exist_ok=True)

        results = {}

        for name in ["cnn_model.keras", "tl_model.keras"]:
            model = tf.keras.models.load_model(os.path.join(ARTIFACTS, name))
            m = self._metrics(model, test_ds)
            results[name] = m

            print(f"\n=== {name} ===")
            print(
                f"accuracy={m['accuracy']}  "
                f"precision={m['precision']}  "
                f"recall={m['recall']}  "
                f"f1={m['f1']}"
            )
            print("confusion_matrix:\n", np.array(m["confusion_matrix"]))

        best = max(results, key=lambda k: results[k]["accuracy"])
        results["best_model"] = best

        print(
            f"\nBest model: {best} "
            f"(accuracy={results[best]['accuracy']})"
        )

        out = os.path.join(ARTIFACTS, "metrics_report.json")

        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        print("Saved report ->", out)

        return results