# src/ml_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
 
from sklearn.model_selection import cross_val_score


class MLModel:

    def __init__(self):
        self.results = []

    # ==========================
    # Train Models
    # ==========================

    def train_decision_tree(self, X_train, y_train):

        model = DecisionTreeClassifier(
            random_state=42
        )

        model.fit(X_train, y_train)

        return model

    def train_random_forest(self, X_train, y_train):

        model = RandomForestClassifier(
            random_state=42
        )

        model.fit(X_train, y_train)

        return model

    def train_svm(self, X_train, y_train):

        model = SVC(
            random_state=42
        )

        model.fit(X_train, y_train)

        return model

    # ==========================
    # Cross Validation
    # ==========================

    def cross_validate(self, model, X_train, y_train):

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="accuracy"
        )

        cv_score = scores.mean()

        print(f"CV Accuracy: {cv_score:.4f}")

        return cv_score

    # ==========================
    # Evaluation
    # ==========================

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # ==========================
    # Confusion Matrix
    # ==========================

    def save_confusion_matrix(
        self,
        model,
        X_test,
        y_test,
        model_name
    ):

        predictions = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            predictions
        )

        plt.figure(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(
            f"{model_name} Confusion Matrix"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig(
            f"reports/{model_name}_confusion_matrix.png"
        )

        plt.close()

    # ==========================
    # Train All Models
    # ==========================

    def train_all_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        models = {
            "Decision Tree":
                self.train_decision_tree(
                    X_train,
                    y_train
                ),

            "Random Forest":
                self.train_random_forest(
                    X_train,
                    y_train
                ),

            "SVM":
                self.train_svm(
                    X_train,
                    y_train
                )
        }

        best_model = None
        best_score = 0

        for model_name, model in models.items():

            print(f"\nTraining {model_name}...")

            cv_score = self.cross_validate(
                model,
                X_train,
                y_train
            )

            metrics = self.evaluate_model(
                model,
                X_test,
                y_test
            )

            self.save_confusion_matrix(
                model,
                X_test,
                y_test,
                model_name.replace(" ", "_")
            )

            self.results.append({
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
                "CV Score": cv_score
            })

            print(f"Accuracy : {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall   : {metrics['recall']:.4f}")
            print(f"F1 Score : {metrics['f1']:.4f}")

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model = model

        self.save_results()

        return best_model, best_score

    # ==========================
    # Save Results
    # ==========================

    def save_results(self):

        results_df = pd.DataFrame(
            self.results
        )

        results_df.to_csv(
            "reports/ml_results.csv",
            index=False
        )

        print(
            "\nResults saved to reports/ml_results.csv"
        )