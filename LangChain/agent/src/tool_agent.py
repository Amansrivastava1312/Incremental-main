import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from Forecaster.scripts import forecast_sales
from mainproj.scripts.detection import vision_chatbot
from nlp_sentiment.distil_bert_test import predict


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    google_api_key=api_key,
)


@tool
def forecast_lookup(
    product_id: str,
    horizon_days: int = 14,
    forecast_method: str = "arima",
):
    """
    Forecast future sales for a product.
    """

    return forecast_sales(
        product_id=product_id,
        horizon_days=horizon_days,
        method=forecast_method,
    )


@tool
def sentiment_lookup(
    review_text: str,
):
    """
    Analyze customer review sentiment.
    """

    return predict(review_text)


@tool
def vision_result_lookup(
    image_path: str,
):
    """
    Analyze image and detect objects.
    """

    image_path = str(
        Path(image_path)
        .expanduser()
        .resolve()
    )

    return vision_chatbot(image_path)


agent = create_agent(
    model=llm,
    tools=[
        forecast_lookup,
        sentiment_lookup,
        vision_result_lookup,
    ],
    system_prompt="""
You are MarketPulse Assistant.

Use:
- forecast_lookup for sales forecasting
- sentiment_lookup for sentiment analysis
- vision_result_lookup for image analysis

Always choose the correct tool and
extract parameters from the user query.
"""
)


