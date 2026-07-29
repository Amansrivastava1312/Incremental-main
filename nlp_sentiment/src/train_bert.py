"""File 2: Run a pretrained BERT-style sentiment pipeline and SAVE it locally.

Uses a 3-class model (negative / neutral / positive) so it matches your
sentiment_label column. Saved with save_pretrained() so it loads offline later.

Run:
    python train_bert.py
"""
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# 3-class sentiment model (negative/neutral/positive)
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SAVE_DIR = "models/bert_sentiment"


def main():
    print(f"[downloading] {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    # quick sanity check
    clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print(clf("The price was fair and delivery was fast"))

    # save model + tokenizer so it can be loaded without internet
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"[saved] BERT sentiment model -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
