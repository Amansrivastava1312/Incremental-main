import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import cross_val_score


class MLModel:

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

        return scores.mean()

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

        return {
            "accuracy": accuracy_score(
                y_test,
                predictions
            ),
            "precision": precision_score(
                y_test,
                predictions,
                average="weighted"
            ),
            "recall": recall_score(
                y_test,
                predictions,
                average="weighted"
            ),
            "f1_score": f1_score(
                y_test,
                predictions,
                average="weighted"
            )
        }

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
            "decision_tree":
                self.train_decision_tree(
                    X_train,
                    y_train
                ),

            "random_forest":
                self.train_random_forest(
                    X_train,
                    y_train
                ),

            "svm":
                self.train_svm(
                    X_train,
                    y_train
                )
        }

        best_model = None
        best_score = 0

        for model_name, model in models.items():

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

            metrics["cv_score"] = cv_score

            joblib.dump(
                {
                    "model": model,
                    "metrics": metrics
                },
                f"reports/{model_name}.pkl"
            )

            if metrics["f1_score"] > best_score:
                best_score = metrics["f1_score"]
                best_model = model

        return best_model, best_score