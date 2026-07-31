import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = f"{MODEL_DIR}/LSTM_GLOBAL.keras"
SCALER_PATH = f"{MODEL_DIR}/LSTM_SCALER.joblib"

LOOKBACK = 30

PRODUCTS = {
    "P001": "PulseBottle Pro",
    "P002": "SonicWave Lite",
    "P003": "FitTrack Neo",
    "P004": "AeroRun Flex",
}


def create_sequences(data, lookback):

    X = []
    y = []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def train_lstm():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    df = pd.read_csv(
        "Forecaster/data/daily_product_sales.csv"
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # FIXED:
    # Fit scaler once on complete dataset
    scaler = MinMaxScaler()

    all_values = (
        df["units_sold"]
        .astype(float)
        .values
        .reshape(-1, 1)
    )

    scaler.fit(all_values)

    all_X = []
    all_y = []

    for product in df["product_id"].unique():

        product_df = (
            df[df["product_id"] == product]
            .sort_values("date")
        )

        values = (
            product_df["units_sold"]
            .astype(float)
            .values
            .reshape(-1, 1)
        )

        scaled_values = scaler.transform(
            values
        )

        X, y = create_sequences(
            scaled_values,
            LOOKBACK,
        )

        all_X.append(X)
        all_y.append(y)

    X_train = np.concatenate(all_X)
    y_train = np.concatenate(all_y)

    model = Sequential(
        [
            LSTM(
                32,
                input_shape=(LOOKBACK, 1),
            ),
            Dense(
                16,
                activation="relu",
            ),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        verbose=1,
    )

    model.save(MODEL_PATH)

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    print(
        "Global LSTM model saved."
    )


def predict_sales(
    product_id,
    days,
):

    product_id = product_id.upper()

    if product_id not in PRODUCTS:
        raise Exception(
            "Invalid Product Id"
        )

    model = load_model(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    df = pd.read_csv(
        "Forecaster/data/daily_product_sales.csv"
    )

    product_df = (
        df[df["product_id"] == product_id]
        .sort_values("date")
    )

    values = (
        product_df["units_sold"]
        .astype(float)
        .values
        .reshape(-1, 1)
    )

    scaled_values = scaler.transform(
        values
    )

    window = (
        scaled_values[-LOOKBACK:]
        .flatten()
        .tolist()
    )

    predictions = []

    for _ in range(days):

        X = np.array(
            window
        ).reshape(
            1,
            LOOKBACK,
            1,
        )

        next_value = float(
            model.predict(
                X,
                verbose=0,
            )[0][0]
        )

        predictions.append(
            next_value
        )

        window.pop(0)
        window.append(
            next_value
        )

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(
            -1,
            1,
        )
    )

    response = {
        "product_id": product_id,
        "product_name": PRODUCTS[
            product_id
        ],
        "horizon_days": days,
        "predicted_sales": [
            round(
                float(x),
                2,
            )
            for x in predictions.flatten()
        ],
    }

    return response


if __name__ == "__main__":

    if (
        not os.path.exists(MODEL_PATH)
        or not os.path.exists(SCALER_PATH)
    ):
        print(
            "Model not found. Training..."
        )
        train_lstm()

    product_id = input(
        "Enter Product Id: "
    )

    days = int(
        input(
            "Enter Forecast Horizon (Days): "
        )
    )

    result = predict_sales(
        product_id,
        days,
    )

    print(result) 