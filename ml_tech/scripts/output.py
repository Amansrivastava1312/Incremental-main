import os
import joblib
import pandas as pd

from tensorflow.keras.models import load_model
from ml_tech.src.feature_engineering import FeatureEngineering


COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def preprocess_input(user_input):

    df = pd.DataFrame(
        [user_input],
        columns=COLUMNS
    )

    fe = FeatureEngineering()

    df = fe.create_features(df)

    preprocessor = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "preprocessor.pkl"
        )
    )

    return preprocessor.transform(df)


def prepare_response(
    prediction,
    metrics
):
    if prediction == 1:
        pred = "Churned"
    else:
        pred = "Not Churned"

    return f"""
    Prediction: {pred}

    Accuracy: {metrics["accuracy"]}

    Precision: {metrics["precision"]}

    Recall: {metrics["recall"]}

    f1_score: {metrics["f1_score"]}
    """


def decision_tree(user_input):

    model = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "decision_tree.pkl"
        )
    )

    metrics = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "decision_tree_metrics.pkl"
        )
    )

    prediction = model.predict(
        preprocess_input(user_input)
    )[0]

    return prepare_response(
        prediction,
        metrics
    )


def random_forest(user_input):

    model = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "random_forest.pkl"
        )
    )

    metrics = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "random_forest_metrics.pkl"
        )
    )

    prediction = model.predict(
        preprocess_input(user_input)
    )[0]

    return prepare_response(
        prediction,
        metrics
    )


def svm(user_input):

    model = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "svm.pkl"
        )
    )

    metrics = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "svm_metrics.pkl"
        )
    )

    prediction = model.predict(
        preprocess_input(user_input)
    )[0]

    return prepare_response(
        prediction,
        metrics
    )


def ann(user_input):

    model = load_model(
        os.path.join(
            BASE_DIR,
            "reports",
            "ann_model.keras"
        )
    )

    prediction = model.predict(
        preprocess_input(user_input),
        verbose=0
    )[0][0]

    pred = 1 if prediction > 0.5 else 0

    metrics = {
        "accuracy": "N/A",
        "precision": "N/A",
        "recall": "N/A",
        "f1_score": "N/A"
    }

    return prepare_response(
        pred,
        metrics
    )