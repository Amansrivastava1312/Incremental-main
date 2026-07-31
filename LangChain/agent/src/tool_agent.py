import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from Forecaster.scripts import forecast_sales
from mainproj.scripts.detection import main as run_vision_detection


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to the .env file."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    google_api_key=api_key,
)


class ToolDecision(BaseModel):
    selected_tool: Literal[
        "forecast_lookup",
        "sentiment_lookup",
        "vision_result_lookup",
    ] = Field(
        description="The tool to use for the question",
    )

    product_id: str | None = Field(
        default=None,
        description=(
            "Product ID mentioned in the question, "
            "such as P100 or PROD123"
        ),
    )

    horizon_days: int | None = Field(
        default=None,
        description=(
            "The number of forecast days, such as 7, 14, or 30"
        ),
    )

    forecast_method: Literal[
        "arima",
        "sarima",
        "lstm",
    ] | None = Field(
        default=None,
        description=(
            "Forecast model to use: arima, sarima, or lstm"
        ),
    )

    image_path: str | None = Field(
        default=None,
        description="Image file path mentioned in the question",
    )


structured_llm = llm.with_structured_output(ToolDecision)


def decide_tool(question: str) -> ToolDecision:
    """Select a tool and extract its inputs from the question."""

    if not question or not question.strip():
        raise ValueError("question is required")

    prompt = f"""
You are a MarketPulse tool router.

Choose exactly one selected_tool from:
- forecast_lookup
- sentiment_lookup
- vision_result_lookup

Tool selection rules:
- Use forecast_lookup for sales forecasts, demand predictions,
  future sales, or future units.
- Use sentiment_lookup for review sentiment, customer opinions,
  and positive or negative review analysis.
- Use vision_result_lookup for image analysis, object detection,
  or image classification.

Extract these values only when present:
- product_id
- horizon_days
- forecast_method
- image_path

forecast_method must be one of:
- arima
- sarima
- lstm

If a value is not present, return null for that value.

Question:
{question}
"""

    return structured_llm.invoke(prompt)


def format_tool_response(
    selected_tool: str,
    tool_input: dict[str, Any],
    tool_output: Any,
) -> dict[str, Any]:
    """Return a common response structure for every tool."""

    return {
        "agent": "marketpulse_router",
        "selected_tool": selected_tool,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "trace": [
            {
                "step": "inspect",
                "detail": (
                    "Matched the request intent to registered tools"
                ),
            },
            {
                "step": "execute",
                "detail": selected_tool,
            },
            {
                "step": "inspect_result",
                "detail": "Tool output returned successfully",
            },
        ],
    }


def forecast_lookup_tool(
    product_id: str | None,
    horizon_days: int = 14,
    forecast_method: str = "arima",
) -> dict[str, Any]:
    """Return a sales forecast for one product."""

    if not product_id or not product_id.strip():
        raise ValueError(
            "product_id is required for forecast lookup"
        )

    if horizon_days < 1:
        raise ValueError(
            "horizon_days must be greater than 0"
        )

    forecast_method = (
        forecast_method or "arima"
    ).strip().lower()

    allowed_methods = {
        "arima",
        "sarima",
        "lstm",
    }

    if forecast_method not in allowed_methods:
        raise ValueError(
            "forecast_method must be arima, sarima, or lstm"
        )

    tool_input = {
        "product_id": product_id,
        "horizon_days": horizon_days,
        "forecast_method": forecast_method,
    }

    tool_output = forecast_sales(
        product_id=product_id,
        horizon_days=horizon_days,
        method=forecast_method,
    )

    return format_tool_response(
        selected_tool="forecast_lookup",
        tool_input=tool_input,
        tool_output=tool_output,
    )


def sentiment_lookup_tool(
    review_text: str | None,
) -> dict[str, Any]:
    """
    Run BERT sentiment prediction in a separate Python process.

    The separate process prevents BERT and Transformers libraries
    from conflicting with libraries loaded in the FastAPI process.
    """

    if not review_text or not review_text.strip():
        raise ValueError(
            "review_text is required for sentiment lookup"
        )

    clean_text = review_text.strip()

    try:
        completed_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "nlp_sentiment.distil_bert_test",
                clean_text,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=str(Path(__file__).resolve().parents[3]),
        )

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "BERT sentiment prediction timed out."
        ) from error

    except OSError as error:
        raise RuntimeError(
            f"Unable to start the BERT subprocess: {error}"
        ) from error

    stderr_output = completed_process.stderr.strip()
    stdout_output = completed_process.stdout.strip()

    if completed_process.returncode != 0:
        raise RuntimeError(
            stderr_output or "BERT sentiment prediction failed."
        )

    if not stdout_output:
        raise RuntimeError(
            "BERT returned an empty sentiment prediction."
        )

    tool_input = {
        "review_text": clean_text,
    }

    tool_output = {
        "sentiment_result": stdout_output,
    }

    return format_tool_response(
        selected_tool="sentiment_lookup",
        tool_input=tool_input,
        tool_output=tool_output,
    )


def vision_result_lookup_tool(
    image=None,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return image classification or object-detection results."""

    if image is None and image_path is None:
        raise ValueError(
            "image or image_path is required for vision lookup"
        )

    if image is not None:
        image_name = (
            getattr(image, "filename", None)
            or getattr(image, "name", None)
            or "uploaded_image"
        )

        tool_input = {
            "image_name": image_name,
        }

        tool_output = run_vision_detection(image)

        return format_tool_response(
            selected_tool="vision_result_lookup",
            tool_input=tool_input,
            tool_output=tool_output,
        )

    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Image path is not a file: {path}"
        )

    tool_input = {
        "image_path": str(path),
    }

    tool_output = run_vision_detection(str(path))

    return format_tool_response(
        selected_tool="vision_result_lookup",
        tool_input=tool_input,
        tool_output=tool_output,
    )


def run_tool_agent(
    question: str,
    product_id: str | None = None,
    horizon_days: int = 14,
    forecast_method: str = "arima",
    image=None,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select and execute the appropriate MarketPulse tool."""

    if not question or not question.strip():
        raise ValueError("question is required")

    clean_question = question.strip()

    # An uploaded image explicitly means that the vision tool
    # should be used. No LLM decision is required.
    if image is not None:
        return vision_result_lookup_tool(
            image=image,
        )

    decision = decide_tool(clean_question)

    selected_product_id = (
        product_id or decision.product_id
    )

    selected_image_path = (
        image_path or decision.image_path
    )

    selected_horizon_days = horizon_days

    if decision.horizon_days is not None:
        selected_horizon_days = decision.horizon_days

    selected_forecast_method = forecast_method

    if decision.forecast_method is not None:
        selected_forecast_method = (
            decision.forecast_method
        )

    if decision.selected_tool == "forecast_lookup":
        return forecast_lookup_tool(
            product_id=selected_product_id,
            horizon_days=selected_horizon_days,
            forecast_method=selected_forecast_method,
        )

    if decision.selected_tool == "sentiment_lookup":
        return sentiment_lookup_tool(
            review_text=clean_question,
        )

    if decision.selected_tool == "vision_result_lookup":
        return vision_result_lookup_tool(
            image_path=selected_image_path,
        )

    raise ValueError(
        "No matching tool was found. "
        "Ask about forecasting, sentiment, or vision."
    )


if __name__ == "__main__":
    question = input(
        "Ask MarketPulse: "
    ).strip()

    image_path_input = input(
        "Image path (press Enter to skip): "
    ).strip()

    result = run_tool_agent(
        question=question,
        image_path=image_path_input or None,
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )