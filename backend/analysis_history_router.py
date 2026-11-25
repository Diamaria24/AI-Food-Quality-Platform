from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.analysis_models import AnalysisHistory
from backend.auth_security import bearer_scheme, get_current_user
from backend.database import get_db


router = APIRouter(
    prefix="/history",
    tags=["Analysis History"],
)


class AnalysisHistoryResponse(BaseModel):
    id: int
    module: str
    analysis_type: str
    food_name: str
    prediction: str
    confidence: float | None
    result_data: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


def current_user_id(
    credentials: Annotated[
        HTTPAuthorizationCredentials,
        Depends(bearer_scheme),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> int:
    user = get_current_user(credentials, db)
    return user.id


@router.get(
    "",
    response_model=list[AnalysisHistoryResponse],
)
def list_history(
    db: Annotated[Session, Depends(get_db)],
    user_id: Annotated[int, Depends(current_user_id)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    statement = (
        select(AnalysisHistory)
        .where(AnalysisHistory.user_id == user_id)
        .order_by(
            AnalysisHistory.created_at.desc(),
            AnalysisHistory.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )

    return db.scalars(statement).all()


@router.get(
    "/{analysis_id}",
    response_model=AnalysisHistoryResponse,
)
def get_history_item(
    analysis_id: int,
    db: Annotated[Session, Depends(get_db)],
    user_id: Annotated[int, Depends(current_user_id)],
):
    record = db.scalar(
        select(AnalysisHistory).where(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == user_id,
        )
    )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found.",
        )

    return record


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_history_item(
    analysis_id: int,
    db: Annotated[Session, Depends(get_db)],
    user_id: Annotated[int, Depends(current_user_id)],
):
    record = db.scalar(
        select(AnalysisHistory).where(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == user_id,
        )
    )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found.",
        )

    db.delete(record)
    db.commit()
