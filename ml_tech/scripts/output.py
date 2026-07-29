#output.py
 
 
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
 
 
import os
 
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
 
def decision_tree(user_input):
 
    model = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "decision_tree.pkl"
        )
    )
 
    data = preprocess_input(user_input)
 
    prediction = model.predict(data)[0]
 
    return "Churned" if prediction == 1 else "Not Churned"
 
 
def random_forest(user_input):
 
    model = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "random_forest.pkl"
        )
    )
 
    data = preprocess_input(user_input)
 
    prediction = model.predict(data)[0]
 
    return "Churned" if prediction == 1 else "Not Churned"
 
 
def svm(user_input):
 
    model = joblib.load(
        os.path.join(
            BASE_DIR,
            "reports",
            "svm.pkl"
        )
    )
 
    data = preprocess_input(user_input)
 
    prediction = model.predict(data)[0]
 
    return "Churned" if prediction == 1 else "Not Churned"
 
 
def ann(user_input):
 
    model = load_model(
        os.path.join(
            BASE_DIR,
            "reports",
            "ann_model.keras"
        )
    )
 
    data = preprocess_input(user_input)
 
    prediction = model.predict(
        data,
        verbose=0
    )[0][0]
 
    return "Churned" if prediction > 0.5 else "Not Churned"