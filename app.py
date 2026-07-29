import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv


# ML
from nlp_sentiment.distil_bert_test import predict as dist_prd
from fastapi import  UploadFile, File

from fastapi.responses import JSONResponse

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

load_dotenv()

app = FastAPI()

# serve static files (css, js, images) from "static" folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# templates folder (same as flask "templates")
templates = Jinja2Templates(directory="templates")


# ---------- INPUT SCHEMA (what frontend sends) ----------
class ChurnInput(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Model: str

# ---------- Forecast----------
class ForecastInput(BaseModel):
    product_id: str
    days: int

# ---------- PAGES ----------
# home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/mlmodel", response_class=HTMLResponse)
def ml_page(request: Request):
    return templates.TemplateResponse(request, "ml.html")

@app.get("/sarima", response_class=HTMLResponse)
def ml_page(request: Request):
    return templates.TemplateResponse(request, "sarima.html")

@app.get("/arima", response_class=HTMLResponse)
def ml_page(request: Request):
    return templates.TemplateResponse(request, "arima.html")

@app.get("/lstm", response_class=HTMLResponse)
def ml_page(request: Request):
    return templates.TemplateResponse(request, "lstm.html")


@app.get("/cnn", response_class=HTMLResponse)
def ml_page(request: Request):
    return templates.TemplateResponse(request, "cnn.html")

@app.get("/transfer", response_class=HTMLResponse)
def ml_page(request: Request):
    return templates.TemplateResponse(request, "transfer.html")

# ---------- API ENDPOINT ----------
@app.post("/predict-ml")
def predict_ml(data: ChurnInput):
    from ml_tech.scripts.output import decision_tree,svm,random_forest

    x=dict(data)
    print(x)
    li = list(x.values())
    dt = li[:-1]
    print(li,dt)
    if li[-1].lower() == "svm":
        result  = svm(dt)
        print(result)
    elif li[-1].lower() == "decision tree":
        result  = decision_tree(dt)
    elif li[-1].lower() == "random forest":
        result  = random_forest(dt)
    else:
        result = "Model Not defined"
        
    
    return {
        "model": data.Model,
        "prediction": result
    }
    



# ---------- API ENDPOINTS ----------
@app.post("/forecast-arima")
def forecast_arima(data: ForecastInput):
    from Forecaster.scripts import forecast_sales

    result = forecast_sales(
        data.product_id,
        data.days,
        "arima"
    )

    return {
        "method": "arima",
        "prediction": result
    }


@app.post("/forecast-lstm")
def forecast_lstm(data: ForecastInput):
    from Forecaster.scripts import forecast_sales

    result = forecast_sales(
        data.product_id,
        data.days,
        "lstm"
    )

    return {
        "method": "lstm",
        "prediction": result
    }


@app.post("/forecast-sarima")
def forecast_sarima(data: ForecastInput):
    from Forecaster.scripts import forecast_sales

    result = forecast_sales(
        data.product_id,
        data.days,
        "sarima"
    )

    return {
        "method": "sarima",
        "prediction": result
    }


@app.post("/forecast-garima")
def forecast_garima(data: ForecastInput):
    from Forecaster.scripts import forecast_sales

    result = forecast_sales(
        data.product_id,
        data.days,
        "garima"
    )

    return {
        "method": "garima",
        "prediction": result
    }

# ---------- INPUT SCHEMA ----------
class SentimentInput(BaseModel):
    text: str



@app.get("/bert", response_class=HTMLResponse)
def bert_page(request: Request):
    return templates.TemplateResponse(request, "bert.html")


@app.get("/logistic", response_class=HTMLResponse)
def logistic_page(request: Request):
    return templates.TemplateResponse(request, "logistic.html")


# ---------- API ENDPOINTS ----------
@app.post("/sentiment-bert")
def sentiment_bert(data: SentimentInput):
    result = dist_prd(data.text.strip())

    return {
        "model": "BERT",
        "sentiment": result
    }


 

@app.post("/predict-cnn")
async def predict_cnn_endpoint(image: UploadFile = File(...)):

    # 1. check the file is a valid image
    if image.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Please upload a valid image (jpg, png, webp)",
            },
        )

    # 2. save the file into uploads folder
    file_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(file_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # 3. run the model
    try:
        prediction = predict_cnn(file_path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
            },
        )

    # 4. return result in the shape the HTML expects
    return {
        "status": "success",
        "method": "cnn",
        "filename": image.filename,
        "prediction": prediction,
    }
    
    
@app.post("/predict-transfer")
async def predict_transfer_endpoint(image: UploadFile = File(...)):

    # 1. check the file is a valid image
    if image.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Please upload a valid image (jpg, png, webp)",
            },
        )

    # 2. save the file into uploads folder
    file_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(file_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # 3. run the transfer learning model
    try:
        prediction = predict_tl(file_path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
            },
        )

    # 4. return result in the shape the HTML expects
    return {
        "status": "success",
        "method": "transfer",
        "filename": image.filename,
        "prediction": prediction,
    }
 