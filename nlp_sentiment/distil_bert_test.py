"""Load the saved BERT sentiment model and expose a predict function.

Import from app.py, or run directly:
    python3 src/run_bert.py "your review text here"
"""
import sys

BERT_DIR = "models/bert_sentiment"

_pipe = None


def load_model():
    """Load the saved HuggingFace pipeline once and cache it."""
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("sentiment-analysis", model=BERT_DIR,
                         tokenizer=BERT_DIR, truncation=True)
    return _pipe


def predict(text):
    """Return (label, score) for a single string."""
    pipe = load_model()
    r = pipe(text)[0]
    return r["label"].lower()
#, round(float(r["score"]), 4)


# if __name__ == "__main__":
#     text = " ".join(sys.argv[1:]) or "Great price and fast delivery"
#     label, score = predict(text)
#     print(f"[bert] {label} ({score})  <-  {text}")
