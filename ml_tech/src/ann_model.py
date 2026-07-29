import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

 
class ANNModel:

    def __init__(self):
        self.model = None

    # ==========================
    # Build ANN
    # ==========================

    def build_model(self, input_dim):

        self.model = Sequential([
            Dense(
                64,
                activation="relu",
                input_shape=(input_dim,)
            ),

            Dense(
                32,
                activation="relu"
            ),

            Dense(
                1,
                activation="sigmoid"
            )
        ])

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return self.model

    # ==========================
    # Train ANN
    # ==========================

    def train_model(
        self,
        X_train,
        y_train
    ):

        history = self.model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        return history

    # ==========================
    # Plot Training Curves
    # ==========================

    def plot_training_curves(self, history):

        # Accuracy Plot
        plt.figure(figsize=(8, 5))

        plt.plot(
            history.history["accuracy"],
            label="Train Accuracy"
        )

        plt.plot(
            history.history["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.title("ANN Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig(
            "reports/ann_accuracy_curve.png"
        )

        plt.close()

        # Loss Plot
        plt.figure(figsize=(8, 5))

        plt.plot(
            history.history["loss"],
            label="Train Loss"
        )

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

        plt.title("ANN Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(
            "reports/ann_loss_curve.png"
        )

        plt.close()

    # ==========================
    # Evaluate ANN
    # ==========================

    def evaluate_model(
        self,
        X_test,
        y_test
    ):

        probabilities = self.model.predict(X_test)

        predictions = (
            probabilities > 0.5
        ).astype(int)

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

        print("\nANN Results")
        print("-" * 30)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        return f1

    # ==========================
    # Save Model
    # ==========================

    def save_model(
        self,
        file_path="reports/ann_model.h5"
    ):

        self.model.save(file_path)

        print(
            f"\nANN Model saved to {file_path}"
        )

    # ==========================
    # Complete Pipeline
    # ==========================

    def run_pipeline(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        self.build_model(
            input_dim=X_train.shape[1]
        )

        history = self.train_model(
            X_train,
            y_train
        )

        self.plot_training_curves(
            history
        )

        score = self.evaluate_model(
            X_test,
            y_test
        )

        self.save_model()

        return self.model, score