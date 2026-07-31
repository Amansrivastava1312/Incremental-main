import os
os.environ["USE_TF"] = "0"

os.environ["TRANSFORMERS_NO_TF"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from LANGRAPH.scripts import run_chatbot
from LangChain.agent.script import run_marketpulse
# from deep_learning.script import predict_cnn,predict_tl

# ML
from mainproj.src.audio_generation import AudioGenerator
from pathlib import Path
from fastapi import  UploadFile, File , Form

from fastapi.responses import JSONResponse
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = "uploads"
STATIC_DIR = PROJECT_ROOT / "static"
AUDIO_OUTPUT_DIR = STATIC_DIR / "audio"

AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

load_dotenv()
import asyncio
import sys
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


 
# ---------- Audio----------
class AudioInput(BaseModel):
    text: str
    language: str = "en"

# ---------- Chatbot----------
class ChatbotInput(BaseModel):
    message: str

# ---------- Forecast----------
class ForecastInput(BaseModel):
    product_id: str
    days: int

# ---------- INPUT SCHEMA ----------
class ImageInput(BaseModel):
    prompt: str
    
class LanggraphChatbotInput(BaseModel):
    question: str

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

@app.get("/imagegen", response_class=HTMLResponse)
def image_page(request: Request):
    return templates.TemplateResponse(request, "imagegen.html")

@app.get("/langraphchatbot", response_class=HTMLResponse)
def chatbot_page(request: Request):
    return templates.TemplateResponse(request, "langraphchatbot.html")


@app.get("/audiogen", response_class=HTMLResponse)
def audio_generation_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="audiogen.html"
    )

@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chatbot.html"
    )

# ---------- LangChain Agent Page ----------

@app.get("/Langagent", response_class=HTMLResponse)
def langchain_agent_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="Langagent.html",
    )


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

# ---------- INPUT SCHEMA ----------
class SentimentInput(BaseModel):
    text: str



@app.get("/bert", response_class=HTMLResponse)
def bert_page(request: Request):
    return templates.TemplateResponse(request, "bert.html")


# @app.get("/logistic", response_class=HTMLResponse)
# def logistic_page(request: Request):
#     return templates.TemplateResponse(request, "logistic.html")



# ---------- API ENDPOINTS ----------
@app.post("/sentiment-bert")
async def sentiment_bert(data: SentimentInput):
    text = data.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "nlp_sentiment.distil_bert_test",
        text,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_message = stderr.decode().strip()

        raise HTTPException(
            status_code=500,
            detail=error_message or "BERT prediction failed."
        )

    result = stdout.decode().strip()

    print("BERT stdout:", repr(result), flush=True)
    print("BERT stderr:", stderr.decode().strip(), flush=True)

    if not result:
        raise HTTPException(
            status_code=500,
            detail="BERT returned an empty prediction."
        )

    return {
        "model": "BERT",
        "sentiment": result
    }


 

@app.post("/predict-cnn")
async def predict_cnn_endpoint(image: UploadFile = File(...)):
    from deep_learning.script import predict_cnn
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
    from deep_learning.script import predict_tl
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
 


# ---------- API ENDPOINT ----------
@app.post("/generate-image")
def generate_image_endpoint(data: ImageInput):
    from mainproj.src.image_generation import ImageGenerator

    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # save into static so browser can load it
    token = os.getenv("HF_TOKEN")
    generator = ImageGenerator(
        hf_token=token,
        output_dir="static/generated"
    )

    try:
        output_path = generator.generate_image(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, watermark, text",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # build url from filename
    filename = os.path.basename(output_path)
    image_url = "/static/generated/" + filename

    return {
        "status": "success",
        "generated_image": image_url
    }


@app.post("/chatbot-response")
def chatbot_response(data: LanggraphChatbotInput):
    question = data.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = run_chatbot(question)

        return {
            "status": "success",
            "answer": result.get("answer", ""),
            "approved": result.get("approved", False),
            "issues": result.get("issues", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/generate-audio")
async def generate_audio_endpoint(data: AudioInput):
    text = data.text.strip()
    language = data.language.strip() or "en"

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    try:
        # Change this import only if your file is elsewhere
        from mainproj.src.audio_generation import AudioGenerator

        generator = AudioGenerator(
            output_dir=str(AUDIO_OUTPUT_DIR)
        )

        output_file = await asyncio.to_thread(
            generator.generate_audio,
            text,
            language
        )

        output_path = Path(output_file)

        if not output_path.exists():
            raise RuntimeError("Audio file was not created.")

        if output_path.stat().st_size == 0:
            raise RuntimeError("Generated audio file is empty.")

        return {
            "status": "success",
            "text": text,
            "language": language,
            "filename": output_path.name,
            "audio_url": f"/static/audio/{output_path.name}"
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print(
            "Audio generation error:",
            repr(error),
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {error}"
        )


@app.post("/chatbot")
async def chatbot_endpoint(data: ChatbotInput):
    question = data.message.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:
        # Lazy import keeps FastAPI startup lightweight
        from chatbot.script import run_chatbot

        # The chatbot call is synchronous and may take time,
        # so run it in a worker thread.
        answer = await asyncio.to_thread(
            run_chatbot,
            question
        )

        if not answer:
            raise RuntimeError(
                "The chatbot returned an empty response."
            )

        return {
            "status": "success",
            "answer": str(answer)
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Chatbot error:",
            repr(error),
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Chatbot failed: {error}"
        )




# ---------- LangChain Agent API ----------

@app.post("/langchain-agent")
async def langchain_agent_endpoint(
    message: str = Form(...),
    image: UploadFile | None = File(None),
):
    question = message.strip()

    if not question and image is None:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    if not question:
        question = "Analyze this image"

    image_path = None

    try:
        if image is not None:
            if image.content_type not in ALLOWED_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Please upload a valid image "
                        "(jpg, png, webp)"
                    ),
                )

            original_name = image.filename or "uploaded_image"
            safe_filename = Path(original_name).name

            file_path = PROJECT_ROOT / UPLOAD_DIR / safe_filename

            content = await image.read()

            with open(file_path, "wb") as file:
                file.write(content)

            image_path = str(file_path)

        result = await asyncio.to_thread(
            run_marketpulse,
            question,
            None,          # product_id
            14,            # horizon_days
            "arima",       # forecast_method
            None,          # image
            image_path,    # image_path
        )

        if isinstance(result, dict):
            tool_output = result.get("tool_output", result)
            answer = str(tool_output)
        else:
            answer = str(result)

        if not answer.strip():
            raise RuntimeError(
                "The LangChain agent returned an empty response."
            )

        return {
            "status": "success",
            "answer": answer,
            "result": result if isinstance(result, dict) else None,
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            "LangChain agent error:",
            repr(error),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"LangChain agent failed: {error}",
        ) from error