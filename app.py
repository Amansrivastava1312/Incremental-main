import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

#Ml
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


# ---------- PAGES ----------
# home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/mlmodel", response_class=HTMLResponse)
def ml_page(request: Request):
    return templates.TemplateResponse(request, "ml.html")


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