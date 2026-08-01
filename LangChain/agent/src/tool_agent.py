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

Always choose the correct tool and extract parameters
from the user query.

IMPORTANT INSTRUCTIONS FOR IMAGE ANALYSIS:

When vision_result_lookup is used, the tool returns data such as:

{
  "image_name": "image_name.jpg",
  "total_objects": 8,
  "object_counts": {
    "long_sleeve_outwear": 6,
    "trousers": 2
  },
  "annotated_image": "/detections/detection/detected_image.jpg",
  "json_file": "/path/to/result.json"
}

After receiving the vision tool response, return ONLY valid JSON
in exactly this structure:

{
  "answer": "Readable summary of detected objects",
  "selected_tool": "vision_result_lookup",
  "tool_input": {},
  "tool_output": {
    "image_name": "exact image_name returned by the tool",
    "total_objects": 0,
    "object_counts": {},
    "annotated_image": "exact annotated_image returned by the tool"
  },
  "annotated_image": "exact annotated_image returned by the tool"
}

Rules:

1. Copy annotated_image exactly from the vision tool response.
2. Do not guess, modify, shorten, or generate an image path.
3. Copy image_name, total_objects, and object_counts exactly.
4. Put annotated_image in both:
   - tool_output.annotated_image
   - top-level annotated_image
5. Return valid JSON only.
6. Do not use Markdown code fences.
7. Do not add any text before or after the JSON.
8. The answer field must contain a readable detection summary.

For forecast and sentiment requests, respond normally.
"""
)
