"""File 3: Load BOTH saved models, run them on the dataset, compare, save CSV.

Requires that you already ran:
    python train_logreg.py
    python train_bert.py

Run:
    python compare_models.py

Outputs:
    artifact/predictions.csv   -> per-review: true label + both model predictions
    artifact/metrics.csv       -> accuracy of each model
"""
import re
import joblib
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

# ---- paths / columns ----
DATA_FILE = "data/competitor_reviews_labeled.csv"
LOGREG_FILE = "models/logreg_model.joblib"
BERT_DIR = "models/bert_sentiment"
PRED_FILE = "artifact/predictions.csv"
METRICS_FILE = "artifact/metrics.csv"
TEXT_COL = "review_text"
LABEL_COL = "sentiment_label"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_FILE).dropna(subset=[TEXT_COL, LABEL_COL])
    texts = df[TEXT_COL].tolist()
    y_true = df[LABEL_COL].str.lower().tolist()

    # ---- Logistic Regression predictions ----
    logreg = joblib.load(LOGREG_FILE)
    logreg_preds = [str(p).lower() for p in logreg.predict([clean_text(t) for t in texts])]

    # ---- BERT predictions ----
    bert = pipeline("sentiment-analysis", model=BERT_DIR, tokenizer=BERT_DIR, truncation=True)
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
