import os
import joblib
import pandas as pd

from tensorflow.keras.models import load_model
from ml_tech.src.feature_engineering import FeatureEngineering


# ==========================================
# Absolute Paths
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)


# ==========================================
# Input Columns
# ==========================================

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


# ==========================================
# Common Preprocessing
# ==========================================

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


# ==========================================
# Decision Tree
# ==========================================

def decision_tree(user_input):

    model = joblib.load(
        os.path.join(
            REPORTS_DIR,
            "decision_tree.pkl"
        )
    )

    data = preprocess_input(user_input)

    prediction = model.predict(data)[0]

    if prediction == 1:
        return "Churned"

    return "Not Churned"


# ==========================================
# Random Forest
# ==========================================

def random_forest(user_input):

    model = joblib.load(
        os.path.join(
            REPORTS_DIR,
            "random_forest.pkl"
        )
    )

    data = preprocess_input(user_input)

    prediction = model.predict(data)[0]

    if prediction == 1:
        return "Churned"

    return "Not Churned"


# ==========================================
# SVM
# ==========================================

def svm(user_input):

    model = joblib.load(
        os.path.join(
            REPORTS_DIR,
            "svm.pkl"
        )
    )

    data = preprocess_input(user_input)

    prediction = model.predict(data)[0]

    if prediction == 1:
        return "Churned"

    return "Not Churned"


# ==========================================
# ANN
# ==========================================

def ann(user_input):

    model = load_model(
        os.path.join(
            REPORTS_DIR,
            "ann_model.keras"
        )
    )

    data = preprocess_input(user_input)

    probability = model.predict(
        data,
        verbose=0
    )[0][0]

    if probability > 0.5:
        return "Churned"

    return "Not Churned"
