from Forecaster.src.arima import predict_sales as arima_predict
from Forecaster.src.lstm import predict_sales as lstm_predict
from Forecaster.src.sarima import predict_sales as sarima_predict

def comparison(product_id, days):
    arima_result = arima_predict(product_id,days)
    lstm_result = lstm_predict(product_id,days)
    sarima_result=sarima_predict(product_id,days)

    arima_prediction=arima_result["predicted_sales"]
    sarima_prediction=sarima_result["predicted_sales"]
    lstm_prediction=lstm_result["predicted_sales"]

    response={
        "arima_prediction":arima_prediction,
        "sarima_prediction":sarima_prediction,
        "lstm_prediction":lstm_prediction
    }
    return response