import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline
from Forecaster.scripts import forecast_sales

load_dotenv()

app = FastAPI()

# templates folder (same as flask "templates")
templates = Jinja2Templates(directory="templates")

# build pipeline once
pipeline = RAGPipeline()


# request body model for /ask
class Question(BaseModel):
    question: str


# request body model for /predict
class ForecastRequest(BaseModel):
    product_id: str
    days: int
    method: str


# home page with 4 buttons
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


# rag chat bot page
@app.get("/rag", response_class=HTMLResponse)
def rag_page(request: Request):
    return templates.TemplateResponse(request, "rag_chatbot.html")


# forecast page
@app.get("/forecast", response_class=HTMLResponse)
def forecast_page(request: Request):
    return templates.TemplateResponse(request, "forecast.html")


# rag answer api
@app.post("/ask")
def ask(data: Question):
    question = data.question

    if not question:
        return {"answer": "Please type a question."}

    answer = pipeline.get_answer(question)
    return {"answer": answer}


# forecast api
@app.post("/predict")
def predict(data: ForecastRequest):
    result = forecast_sales(data.product_id, data.days, data.method)
    producName = result["data"]["product_name"]
    return {"result": str(producName)}