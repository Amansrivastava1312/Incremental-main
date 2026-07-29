import os
import joblib
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA 
def train_arima():
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_dir=BASE_DIR / "models"
    os.makedirs(model_dir,exist_ok=True)

    df =pd.read_csv("Forecaster/data/daily_product_sales.csv")

    df["date"]=pd.to_datetime(df["date"])

    for product in df["product_id"].unique():
        product_df=(df[df["product_id"]==product].sort_values("date"))
        ts=product_df["units_sold"]
        ts = product_df.set_index("date")["units_sold"].asfreq("D")
        ts = ts.interpolate()

        model=ARIMA(ts,order=(5,1,3))
        model_fit=model.fit()

        with open(f"{model_dir}/ARIMA_{product}.joblib","wb") as f:
            joblib.dump(model_fit, f)

        print(f"{product} model saved.")

model="models"

def predict_sales(product_id,days):

    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "models" / f"ARIMA_{product_id.upper()}.joblib"
    with open(model_path,"rb") as f:
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
    train_arima()
