import json
import math
from typing import Any

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from backend.analysis_models import AnalysisHistory


def _text(value: Any, fallback: str, max_length: int) -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    return text[:max_length] if text else fallback


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    if not 0 <= number <= 1:
        return None

    return number


def _prediction(result: dict, fallback: str) -> str:
    nested = result.get("result")

    if isinstance(nested, dict):
        for key in ("prediction", "predicted_class", "label"):
            value = nested.get(key)
            if value is not None:
                return _text(value, fallback, 255)

    for key in ("prediction", "predicted_class", "label"):
        value = result.get(key)
        if value is not None:
            return _text(value, fallback, 255)

    return fallback


def _extract_confidence(result: dict) -> float | None:
    nested = result.get("result")

    if isinstance(nested, dict):
        for key in ("confidence", "confidence_score"):
            if key in nested:
                return _confidence(nested[key])

    for key in ("confidence", "confidence_score"):
        if key in result:
            return _confidence(result[key])

    return None


def save_analysis_history(
    *,
    db: Session,
    user_id: int,
    module: str,
    analysis_type: str,
    food_name: str,
    result: dict,
    prediction: str | None = None,
    confidence: float | None = None,
) -> AnalysisHistory:
    if not isinstance(result, dict):
        raise ValueError("Analysis result must be a dictionary.")

    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("A valid authenticated user ID is required.")

    encoded = jsonable_encoder(result)

    # Validate that the complete result can be stored as JSON.
    json.dumps(encoded, allow_nan=False)

    label = (
        _text(prediction, "Analysis completed", 255)
        if prediction is not None
        else _prediction(encoded, "Analysis completed")
    )

    score = (
        _confidence(confidence)
        if confidence is not None
        else _extract_confidence(encoded)
    )

    record = AnalysisHistory(
        user_id=user_id,
        module=_text(module, "unknown", 50),
        analysis_type=_text(analysis_type, "analysis", 100),
        food_name=_text(food_name, "Unknown", 150),
        prediction=label,
        confidence=score,
        result_data=encoded,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record
