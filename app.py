import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from Forecaster.scripts import forecast_sales




from rag_chatbot.script import run_chatbot
from ml_tech.scripts.output import decision_tree,svm,random_forest

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


# ---------- API ENDPOINT ----------
@app.post("/predict-ml")
def predict_ml(data: ChurnInput):
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
    result = forecast_sales(data.product_id, data.days, "arima")
    return {
        "method": "arima",
        "prediction": result
    }
    
@app.post("/forecast-lstm")
def forecast_arima(data: ForecastInput):
    result = forecast_sales(data.product_id, data.days, "lstm")
    return {
        "method": "lstm",
        "prediction": result
    }


@app.post("/forecast-sarima")
def forecast_sarima(data: ForecastInput):
    result = forecast_sales(data.product_id, data.days, "sarima")
    return {
        "method": "sarima",
        "prediction": result
    }


@app.post("/forecast-garima")
def forecast_garima(data: ForecastInput):
    result = forecast_sales(data.product_id, data.days, "garima")
    return {
        "method": "garima",
        "prediction": result
    }
    


# request body for chatbot
class ChatInput(BaseModel):
    question: str

# serve the chatbot page
@app.get("/rag-chatbot", response_class=HTMLResponse)
def agent_page(request: Request):
    return templates.TemplateResponse(request, "rag-chatbot.html")

# chatbot answer endpoint
@app.post("/rag-chatbot")
def chat(data: ChatInput):
    try:
        answer = run_chatbot(data.question)
        return {
            "status": "success",
            "question": data.question,
            "answer": answer
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
