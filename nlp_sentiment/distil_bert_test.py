import os
import sys
from pathlib import Path
from transformers import pipeline

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def predict(text):
    clf = pipeline("sentiment-analysis", model=MODEL_ID, tokenizer=MODEL_ID)
    result = clf(text)
    return result[0]["label"].title()


def main():
    question = input("\nType a review\n")
    answer = predict(question)
    print(f"The review's sentiment is :{answer}")

if __name__ == "__main__":
    main()