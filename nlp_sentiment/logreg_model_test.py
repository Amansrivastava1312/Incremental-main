"""Load the saved Logistic Regression model and expose a predict function.

Import from app.py, or run directly:
    python3 src/run_logreg.py "your review text here"
"""
import os
import re
import sys
import joblib

# ---- project root (one level above this file: src/ -> nlp_sentiment/) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE = os.path.join(ROOT, "models", "logreg_model.joblib")

_model = None


def _clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model():
    """Load the saved pipeline once and cache it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_FILE)
    return _model


def predict(text):
    """Return the sentiment label for a single string."""
    model = load_model()
    return str(model.predict([_clean(text)])[0])


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Great price and fast delivery"
    print(f"[logreg] {predict(text)}  <-  {text}")