from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from backend.auth_passwords import password_hasher
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.user_models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])




class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=150)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("full_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()

        if len(value) < 2:
            raise ValueError("Name must contain at least 2 characters.")

        return value

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    created_at: datetime

    model_config = {"from_attributes": True}


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def register_user(
    payload: RegisterRequest,
    db: Annotated[Session, Depends(get_db)]
):
    existing = db.scalar(
        select(User).where(User.email == str(payload.email))
    )

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists."
        )

    user = User(
        full_name=payload.full_name,
        email=str(payload.email),
        hashed_password=password_hasher.hash(payload.password)
    )

    db.add(user)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists."
        )

    db.refresh(user)

    return user


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
def login_user(
    payload: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
):
    from backend.auth_security import (
        authenticate_user,
        create_access_token,
    )

    user = authenticate_user(
        db,
        str(payload.email),
        payload.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    return TokenResponse(
        access_token=create_access_token(user.id)
    )


@router.get("/me", response_model=UserResponse)
def get_my_profile(
    db: Annotated[Session, Depends(get_db)],
    credentials=Depends(
        __import__(
            "backend.auth_security",
            fromlist=["bearer_scheme"]
        ).bearer_scheme
    ),
):
    from backend.auth_security import get_current_user

    user = get_current_user(credentials, db)
    return user
