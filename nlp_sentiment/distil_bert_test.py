import os
import sys
from pathlib import Path

# Configure native libraries before importing Transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BERT_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "bert_sentiment"
)

_pipe = None


def load_model():
    global _pipe

    if _pipe is None:
        print(
            f"[BERT] Loading model from: {BERT_DIR}",
            file=sys.stderr,
            flush=True
        )

        if not BERT_DIR.exists():
            raise FileNotFoundError(
                f"BERT model directory not found: {BERT_DIR}"
            )

        # Lazy import: import only when BERT is requested
        from transformers import pipeline

        print(
            "[BERT] Creating PyTorch pipeline...",
            file=sys.stderr,
            flush=True
        )

        _pipe = pipeline(
            task="sentiment-analysis",
            model=str(BERT_DIR),
            tokenizer=str(BERT_DIR),
            framework="pt",
            device=-1
        )

        print(
            "[BERT] Model loaded successfully.",
            file=sys.stderr,
            flush=True
        )

    return _pipe


def predict(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text cannot be empty.")

    pipe = load_model()

    print("[BERT] Running inference...", 
          file=sys.stderr,
          flush=True)

    result = pipe(
        text.strip(),
        truncation=True,
        max_length=512
    )[0]

    print(
        f"[BERT] Raw prediction: {result}",
        file=sys.stderr,
        flush=True
    )

    return result["label"].lower()


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]).strip()

    if not text:
        print("Text cannot be empty.", file=sys.stderr)
        sys.exit(1)

    prediction = predict(text)

    # This is captured by app.py as stdout
    print(prediction, flush=True)