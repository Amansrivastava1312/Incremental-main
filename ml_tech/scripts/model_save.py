import joblib

from src.data_cleaning import DataCleaning
from src.feature_engineering import FeatureEngineering
from src.ml_model import MLModel
from src.ann_model import ANNModel


def train_and_save_models():

    cleaner = DataCleaning(
        file_path="data/Churn_Modelling.csv",
        target_column="Exited"
    )

    df = cleaner.load_data()

    fe = FeatureEngineering()

    df = fe.create_features(df)
    df = fe.remove_correlated_features(df)

    X_train, X_test, y_train, y_test = cleaner.split_data(df)

    preprocessor = cleaner.create_preprocessor(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    joblib.dump(
        preprocessor,
        "reports/preprocessor.pkl"
    )

    ml = MLModel()

    dt_model = ml.train_decision_tree(
        X_train,
        y_train
    )

    rf_model = ml.train_random_forest(
        X_train,
        y_train
    )

    svm_model = ml.train_svm(
        X_train,
        y_train
    )

    joblib.dump(
        dt_model,
        "reports/decision_tree.pkl"
    )

    joblib.dump(
        rf_model,
        "reports/random_forest.pkl"
    )

    joblib.dump(
        svm_model,
        "reports/svm.pkl"
    )

    ann = ANNModel()

    ann.build_model(
        X_train.shape[1]
    )

    ann.train_model(
        X_train,
        y_train
    )

    ann.save_model(
        "reports/ann_model.keras"
    )

    print("All models saved successfully!")


if __name__ == "__main__":
    train_and_save_models()