"""File 3: Load BOTH saved models, run them on the dataset, compare, save CSV.

Requires that you already ran:
    python train_logreg.py

Run:
    python compare_models.py

Outputs:
    artifact/predictions.csv   -> per-review: true label + both model predictions
    artifact/metrics.csv       -> accuracy of each model
"""
import os
import re
import joblib
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

# ---- project root (one level above this file: src/ -> nlp_sentiment/) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- paths / columns ----
DATA_FILE = os.path.join(ROOT, "data", "competitor_reviews_labeled.csv")
LOGREG_FILE = os.path.join(ROOT, "models", "logreg_model.joblib")
PRED_FILE = os.path.join(ROOT, "artifact", "predictions.csv")
METRICS_FILE = os.path.join(ROOT, "artifact", "metrics.csv")
TEXT_COL = "review_text"
LABEL_COL = "sentiment_label"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # make sure the output folder exists
    os.makedirs(os.path.join(ROOT, "artifact"), exist_ok=True)

    df = pd.read_csv(DATA_FILE).dropna(subset=[TEXT_COL, LABEL_COL])
    texts = df[TEXT_COL].tolist()
    y_true = df[LABEL_COL].str.lower().tolist()

    # ---- Logistic Regression predictions ----
    logreg = joblib.load(LOGREG_FILE)
    logreg_preds = [str(p).lower() for p in logreg.predict([clean_text(t) for t in texts])]

    # ---- BERT predictions ----
    MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    bert = pipeline("sentiment-analysis", model=MODEL_ID, tokenizer=MODEL_ID, truncation=True)
    bert_preds = [r["label"].lower() for r in bert(texts)]

    # ---- per-review comparison table ----
    out = pd.DataFrame({
        "review_text": texts,
        "true_label": y_true,
        "logreg_pred": logreg_preds,
        "bert_pred": bert_preds,
    })
    out["logreg_correct"] = out["true_label"] == out["logreg_pred"]
    out["bert_correct"] = out["true_label"] == out["bert_pred"]
    out.to_csv(PRED_FILE, index=False)
    print(f"[saved] per-review predictions -> {PRED_FILE}")

    # ---- metrics summary ----
    metrics = pd.DataFrame({
        "model": ["logistic_regression", "bert"],
        "accuracy": [
            round(accuracy_score(y_true, logreg_preds), 4),
            round(accuracy_score(y_true, bert_preds), 4),
        ],
        "n_samples": [len(y_true), len(y_true)],
    })
    metrics.to_csv(METRICS_FILE, index=False)
    print(f"[saved] metrics summary   -> {METRICS_FILE}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()