import os
import joblib
import pandas as pd

from tensorflow.keras.models import load_model
from src.feature_engineering import FeatureEngineering


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)


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


MODEL_SCORES = {
    "Decision Tree": 0.7939,
    "Random Forest": 0.8515,
    "SVM": 0.8428,
    "ANN": 0.8512
}


def preprocess_input(user_input):

    df = pd.DataFrame(
        [user_input],
        columns=COLUMNS
    )

    fe = FeatureEngineering()

    df = fe.create_features(df)

    preprocessor = joblib.load(
        os.path.join(
            REPORTS_DIR,
            "preprocessor.pkl"
        )
    )

    return preprocessor.transform(df)


def compare_models(user_input):

    data = preprocess_input(user_input)

    # Decision Tree

    dt_model = joblib.load(
        os.path.join(
            REPORTS_DIR,
            "decision_tree.pkl"
        )
    )

    dt_prediction = (
        "Churned"
        if dt_model.predict(data)[0] == 1
        else "Not Churned"
    )

    # Random Forest

    rf_model = joblib.load(
        os.path.join(
            REPORTS_DIR,
            "random_forest.pkl"
        )
    )

    rf_prediction = (
        "Churned"
        if rf_model.predict(data)[0] == 1
        else "Not Churned"
    )

    # SVM

    svm_model = joblib.load(
        os.path.join(
            REPORTS_DIR,
            "svm.pkl"
        )
    )

    svm_prediction = (
        "Churned"
        if svm_model.predict(data)[0] == 1
        else "Not Churned"
    )

    # ANN

    ann_model = load_model(
        os.path.join(
            REPORTS_DIR,
            "ann_model.keras"
        )
    )

    ann_prob = ann_model.predict(
        data,
        verbose=0
    )[0][0]

    ann_prediction = (
        "Churned"
        if ann_prob > 0.5
        else "Not Churned"
    )

    report = {
        "Decision Tree": {
            "Prediction": dt_prediction,
            "F1 Score": MODEL_SCORES["Decision Tree"]
        },

        "Random Forest": {
            "Prediction": rf_prediction,
            "F1 Score": MODEL_SCORES["Random Forest"]
        },

        "SVM": {
            "Prediction": svm_prediction,
            "F1 Score": MODEL_SCORES["SVM"]
        },

        "ANN": {
            "Prediction": ann_prediction,
            "F1 Score": MODEL_SCORES["ANN"]
        },

        "Best Model": "Random Forest"
    }

    return report