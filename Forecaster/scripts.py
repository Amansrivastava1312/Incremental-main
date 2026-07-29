"""
Forecast Orchestrator

Supported Models:
- ARIMA
- SARIMA
- LSTM

Receives requests from frontend and returns
a standardized response for downstream agents.
"""

from typing import Dict, Any

from Forecaster.src.arima import predict_sales as arima_predict
from Forecaster.src.sarima import predict_sales as sarima_predict
from Forecaster.src.lstm import predict_sales as lstm_predict
from Forecaster.src.comparison import comparison 

SUPPORTED_MODELS = ["arima", "sarima", "lstm"]


def forecast_sales(
    product_id: str,
    horizon_days: int,
    method: str,
) -> Dict[str, Any]:

    method = method.lower()

    if method not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    if horizon_days < 1:
        raise ValueError("horizon_days must be greater than 0")

    if method == "arima":
        result = arima_predict(product_id, horizon_days)

    elif method == "sarima":
        result = sarima_predict(product_id, horizon_days)

    elif method == "lstm":
        result = lstm_predict(product_id, horizon_days)

    return {
        "status": "success",
        "method": method,
        "data": result
    }

def compare(product_id,horizon_days):
    result=comparison(product_id,horizon_days)
    return result

if __name__ == "__main__":

    product_id = input("Enter Product ID: ")
    horizon_days = int(input("Enter Forecast Horizon (Days): "))
    # method = input("Enter Model (arima/sarima/lstm): ")

    # response = forecast_sales(
    #     product_id=product_id,
    #     horizon_days=horizon_days,
    #     method=method,
    # )
    response=compare(product_id,horizon_days)
    print(response)