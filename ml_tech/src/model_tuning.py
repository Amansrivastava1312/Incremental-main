import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import f1_score


class ModelTuning:

    # =====================================
    # Learning Curve
    # =====================================

    def plot_learning_curve(
        self,
        model,
        X_train,
        y_train,
        model_name
    ):

        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="accuracy"
        )

        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)

        plt.figure(figsize=(8, 5))

        plt.plot(
            train_sizes,
            train_mean,
            label="Training Score"
        )

        plt.plot(
            train_sizes,
            test_mean,
            label="Validation Score"
        )

        plt.title(f"{model_name} Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig(
            f"reports/{model_name}_learning_curve.png"
        )

        plt.close()

    # =====================================
    # Check Class Imbalance
    # =====================================

    def check_class_imbalance(self, y_train):

        print("\nClass Distribution:")
        print(y_train.value_counts(normalize=True))

    # =====================================
    # Tune Random Forest
    # =====================================

    def tune_random_forest(
        self,
        X_train,
        y_train
    ):

        model = RandomForestClassifier(
            random_state=42
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None]
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="f1_weighted",
            n_jobs=-1
        )

        grid_search.fit(
            X_train,
            y_train
        )

        print("\nRandom Forest Best Parameters:")
        print(grid_search.best_params_)

        return grid_search.best_estimator_

    # =====================================
    # Tune SVM
    # =====================================

    def tune_svm(
        self,
        X_train,
        y_train
    ):

        model = SVC()

        param_dist = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=4,
            cv=3,
            scoring="f1_weighted",
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(
            X_train,
            y_train
        )

        print("\nSVM Best Parameters:")
        print(random_search.best_params_)

        return random_search.best_estimator_

    # =====================================
    # Evaluate Tuned Model
    # =====================================

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        predictions = model.predict(X_test)

        return f1_score(
            y_test,
            predictions,
            average="weighted"
        )

    # =====================================
    # Save Best Model
    # =====================================

    def save_best_model(
        self,
        model,
        file_name="reports/best_model.pkl"
    ):

        joblib.dump(
            model,
            file_name
        )

        print(
            f"\nBest model saved to {file_name}"
        )

    # =====================================
    # Tune All Models
    # =====================================

    def tune_all_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        self.check_class_imbalance(
            y_train
        )

        rf_model = self.tune_random_forest(
            X_train,
            y_train
        )

        svm_model = self.tune_svm(
            X_train,
            y_train
        )

        self.plot_learning_curve(
            rf_model,
            X_train,
            y_train,
            "RandomForest"
        )

        self.plot_learning_curve(
            svm_model,
            X_train,
            y_train,
            "SVM"
        )

        rf_score = self.evaluate_model(
            rf_model,
            X_test,
            y_test
        )

        svm_score = self.evaluate_model(
            svm_model,
            X_test,
            y_test
        )

        print(
            f"\nRandom Forest F1 Score: {rf_score:.4f}"
        )

        print(
            f"SVM F1 Score: {svm_score:.4f}"
        )

        if rf_score > svm_score:
            best_model = rf_model
            best_score = rf_score
            best_name = "Random Forest"
        else:
            best_model = svm_model
            best_score = svm_score
            best_name = "SVM"

        self.save_best_model(best_model)

        print(
            f"\nBest Tuned Model: {best_name}"
        )

        return best_model, best_score 