from pathlib import Path

from .src.tool_agent import (
    forecast_lookup_tool,
    run_tool_agent,
    sentiment_lookup_tool,
    vision_result_lookup_tool,
)


def run_forecast(
    product_id: str,
    horizon_days: int = 14,
    forecast_method: str = "arima",
):
    return forecast_lookup_tool(
        product_id=product_id,
        horizon_days=horizon_days,
        forecast_method=forecast_method,
    )


def run_sentiment(review_text: str):
    return sentiment_lookup_tool(
        review_text=review_text,
    )


def run_vision(
    image=None,
    image_path: str | Path | None = None,
):
    return vision_result_lookup_tool(
        image=image,
        image_path=image_path,
    )


def run_marketpulse(
    question: str,
    product_id: str | None = None,
    horizon_days: int = 14,
    forecast_method: str = "arima",
    image=None,
    image_path: str | Path | None = None,
):
    return run_tool_agent(
        question=question,
        product_id=product_id,
        horizon_days=horizon_days,
        forecast_method=forecast_method,
        image=image,
        image_path=image_path,
    )