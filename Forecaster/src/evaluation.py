import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model


ARTIFACT_DIR = "artifacts/forecasting"

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True,
)

PRODUCTS = [
    "P001",
    "P002",
    "P003",
    "P004",
]

LOOKBACK = 30


def evaluate():

    results = {}

    df = pd.read_csv(
        "data/daily_product_sales.csv"
    )

    lstm_model = load_model(
        "models/LSTM_GLOBAL.keras"
    )

    lstm_scaler = joblib.load(
        "models/LSTM_SCALER.joblib"
    )

    for product in PRODUCTS:

        product_df = (
            df[df["product_id"] == product]
            .sort_values("date")
        )

        values = (
            product_df["units_sold"]
            .astype(float)
            .values
        )

        split = int(
            len(values) * 0.8
        )

        train = values[:split]
        test = values[split:]

        # ---------------------------------
        # ARIMA
        # ---------------------------------

        arima_model = joblib.load(
            f"models/ARIMA_{product}.joblib"
        )

        arima_forecast = arima_model.forecast(
            steps=len(test)
        )

        arima_mse = mean_squared_error(
            test,
            arima_forecast,
        )

        # ---------------------------------
        # SARIMA
        # ---------------------------------

        sarima_model = joblib.load(
            f"models/SARIMA_{product}.joblib"
        )

        sarima_forecast = sarima_model.forecast(
            steps=len(test)
        )

        sarima_mse = mean_squared_error(
            test,
            sarima_forecast,
        )

        # ---------------------------------
        # LSTM
        # ---------------------------------

        scaled_values = lstm_scaler.transform(
            values.reshape(-1, 1)
        )

        lstm_actual = []
        lstm_predicted = []

        for i in range(
            LOOKBACK,
            len(scaled_values)
        ):
            x = scaled_values[
                i - LOOKBACK:i
            ].reshape(
                1,
                LOOKBACK,
                1,
            )

            prediction = float(
                lstm_model.predict(
                    x,
                    verbose=0,
                )[0][0]
            )

            lstm_predicted.append(
                prediction
            )

            lstm_actual.append(
                scaled_values[i][0]
            )

        lstm_actual = (
            lstm_scaler.inverse_transform(
                np.array(
                    lstm_actual
                ).reshape(-1, 1)
            )
            .flatten()
        )

        lstm_predicted = (
            lstm_scaler.inverse_transform(
                np.array(
                    lstm_predicted
                ).reshape(-1, 1)
            )
            .flatten()
        )

        lstm_mse = mean_squared_error(
            lstm_actual,
            lstm_predicted,
        )

        # ---------------------------------
        # RESULTS
        # ---------------------------------

        results[product] = {
            "ARIMA_MSE": round(
                float(arima_mse),
                2,
            ),
            "SARIMA_MSE": round(
                float(sarima_mse),
                2,
            ),
            "LSTM_MSE": round(
                float(lstm_mse),
                2,
            ),
        }

        # ---------------------------------
        # COMPARISON PLOT
        # ---------------------------------

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            test,
            label="Actual",
            linewidth=2,
        )

        plt.plot(
            arima_forecast,
            label="ARIMA",
        )

        plt.plot(
            sarima_forecast,
            label="SARIMA",
        )

        plt.legend()

        plt.title(
            f"{product} Forecast Comparison"
        )

        plt.savefig(
            f"{ARTIFACT_DIR}/{product}_comparison.png"
        )

        plt.close()

    # ---------------------------------
    # SAVE METRICS JSON
    # ---------------------------------

    with open(
        f"{ARTIFACT_DIR}/metrics.json",
        "w",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # ---------------------------------
    # GENERATE REPORT
    # ---------------------------------

    with open(
        f"{ARTIFACT_DIR}/forecasting_report.md",
        "w",
    ) as file:

        file.write(
            "# Forecasting Evaluation Report\n\n"
        )

        for product, metrics in results.items():

            file.write(
                f"## {product}\n"
            )

            file.write(
                f"- ARIMA MSE: {metrics['ARIMA_MSE']}\n"
            )

            file.write(
                f"- SARIMA MSE: {metrics['SARIMA_MSE']}\n"
            )

            file.write(
                f"- LSTM MSE: {metrics['LSTM_MSE']}\n\n"
            )

    print(
        "Forecasting evaluation completed."
    )


if __name__ == "__main__":
    evaluate()