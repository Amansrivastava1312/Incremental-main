import os 
import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
model_dir=BASE_DIR / "models"
def train_sarima():
    os.makedirs(model_dir,exist_ok=True)
    df=pd.read_csv("Forecaster/data/daily_product_sales.csv")
    df["date"]=pd.to_datetime(df["date"])
    for product in df["product_id"].unique():
        product_df=(df[df["product_id"]==product].sort_values("date"))

        ts=product_df["units_sold"]
        model=SARIMAX(ts,order=(1,1,1),
                      seasonal_order=(1,1,1,7),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        model_fit=model.fit(disp=False)
        with open(f"{model_dir}/SARIMA_{product}.joblib","wb") as f:
            joblib.dump(model_fit,f)
        print(f"{product} SARIMA model saved")

def predict_sales(product_id,days):
    with open(f"{model_dir}/SARIMA_{product_id.upper()}.joblib","rb") as f:
        model=joblib.load(f)

    if product_id.upper()=="P001":
        product_name="PulseBottle Pro"
    elif product_id.upper()=="P002":
        product_name="SonicWave Lite"
    elif product_id.upper()=="P003":
        product_name="FitTrack Neo"
    elif product_id.upper()=="P004":
        product_name="AeroRun Flex"
    else:
        raise Exception("Product Id is invalid")

    forecast=model.forecast(steps=days)
    response={
        "product_id":product_id,
        "product_name":product_name,
        "horizon_days":days,
        "predicted_sales":[round(x,2) for x in forecast]
    }
    return response

if __name__=="__main__":
    # product_id=input("Enter product id ")
    # days=int(input("enter days"))
    # print(predict_sales(product_id,days))
    train_sarima()