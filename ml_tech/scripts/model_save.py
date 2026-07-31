import joblib

from ml_tech.src.data_cleaning import DataCleaning
from ml_tech.src.feature_engineering import FeatureEngineering
from ml_tech.src.ml_model import MLModel
from ml_tech.src.ann_model import ANNModel


def train_and_save_models():

    cleaner = DataCleaning(
        file_path="ml_tech/data/Churn_Modelling.csv",
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
        "ml_tech/reports/preprocessor.pkl"
    )

    ml = MLModel()

    # ==========================
    # Train Models
    # ==========================

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

    # ==========================
    # Evaluate Models
    # ==========================

    dt_metrics = ml.evaluate_model(
        dt_model,
        X_test,
        y_test
    )

    rf_metrics = ml.evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    svm_metrics = ml.evaluate_model(
        svm_model,
        X_test,
        y_test
    )

    # ==========================
    # Save Models
    # ==========================

    joblib.dump(
        dt_model,
        "ml_tech/reports/decision_tree.pkl"
    )

    joblib.dump(
        rf_model,
        "ml_tech/reports/random_forest.pkl"
    )

    joblib.dump(
        svm_model,
        "ml_tech/reports/svm.pkl"
    )

    # ==========================
    # Save Metrics
    # ==========================

    joblib.dump(
        dt_metrics,
        "ml_tech/reports/decision_tree_metrics.pkl"
    )

    joblib.dump(
        rf_metrics,
        "ml_tech/reports/random_forest_metrics.pkl"
    )

    joblib.dump(
        svm_metrics,
        "ml_tech/reports/svm_metrics.pkl"
    )

    # ==========================
    # ANN
    # ==========================

    ann = ANNModel()

    ann.build_model(
        X_train.shape[1]
    )

    ann.train_model(
        X_train,
        y_train
    )

    ann.save_model(
        "ml_tech/reports/ann_model.keras"
    )

    print("All models and metrics saved successfully!")


if __name__ == "__main__":
    train_and_save_models()