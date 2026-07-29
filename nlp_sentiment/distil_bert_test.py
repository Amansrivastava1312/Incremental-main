import os
from pathlib import Path

# Configure native libraries before importing Transformers
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
            flush=True
        )

    return _pipe


def predict(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text cannot be empty.")

    pipe = load_model()

    print("[BERT] Running inference...", flush=True)

    result = pipe(
        text.strip(),
        truncation=True,
        max_length=512
    )[0]

    print(
        f"[BERT] Raw prediction: {result}",
        flush=True
    )

    return result["label"].lower()