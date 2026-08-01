"""File 1: Train a simple Logistic Regression sentiment model and SAVE it.

Run:
    python train_logreg.py
"""
import os
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---- project root (one level above this file: src/ -> nlp_sentiment/) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- paths / columns ----
DATA_FILE = os.path.join(ROOT, "data", "competitor_reviews_labeled.csv")
MODEL_FILE = os.path.join(ROOT, "models", "logreg_model.joblib")
TEXT_COL = "review_text"
LABEL_COL = "sentiment_label"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # make sure the models folder exists before saving
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    df = pd.read_csv(DATA_FILE).dropna(subset=[TEXT_COL, LABEL_COL])
    X = df[TEXT_COL].apply(clean_text)
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)

    print(classification_report(y_test, pipe.predict(X_test), zero_division=0))

    joblib.dump(pipe, MODEL_FILE)
    print(f"[saved] logistic regression model -> {MODEL_FILE}")


if __name__ == "__main__":
    main()