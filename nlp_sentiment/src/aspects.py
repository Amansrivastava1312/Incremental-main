
import re
from collections import Counter
import pandas as pd

# ---- paths ----
PRED_FILE = "artifact/predictions.csv"
OUT_FILE = "artifact/aspect_sentiment.csv"

# ---- Task 2: simple aspect keywords ----
ASPECT_KEYWORDS = {
    "price":    ["price", "cost", "expensive", "cheap", "overpriced", "rupee",
                 "money", "value", "affordable", "worth"],
    "service":  ["service", "support", "staff", "customer", "help", "refund",
                 "rude", "polite", "response"],
    "quality":  ["quality", "material", "build", "broke", "damaged", "flimsy",
                 "works", "reliable", "durable", "sturdy"],
    "delivery": ["delivery", "shipping", "arrived", "late", "fast", "packaging",
                 "order", "shipment", "dispatch"],
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_aspects(text):
    """Return list of aspects mentioned in a single review."""
    words = set(clean_text(text).split())
    return [aspect for aspect, kws in ASPECT_KEYWORDS.items()
            if words.intersection(kws)]


def main():
    df = pd.read_csv(PRED_FILE)

    # which sentiment column to use: prefer the true label
    label_col = "true_label" if "true_label" in df.columns else "bert_pred"

    # Task 1 + 3: count each aspect per sentiment
    table = {a: Counter() for a in ASPECT_KEYWORDS}
    for _, row in df.iterrows():
        for aspect in extract_aspects(row["review_text"]):
            table[aspect][row[label_col]] += 1

    # build a tidy DataFrame
    out = pd.DataFrame(table).T.fillna(0).astype(int)
    out.index.name = "aspect"
    out["total"] = out.sum(axis=1)
    out = out.sort_values("total", ascending=False)

    out.to_csv(OUT_FILE)
    print(f"[saved] aspect-sentiment counts -> {OUT_FILE}")
    print(out.to_string())


if __name__ == "__main__":
    main()
