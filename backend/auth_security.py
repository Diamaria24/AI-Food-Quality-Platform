import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.user_models import User
from backend.auth_passwords import password_hasher

load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing from .env")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 60

bearer_scheme = HTTPBearer(auto_error=False)


def authenticate_user(db: Session, email: str, password: str):
    from sqlalchemy import select

    user = db.scalar(
        select(User).where(User.email == email.strip().lower())
    )

    if user is None:
        return None

    if not password_hasher.verify(password, user.hashed_password):
        return None

    return user


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_MINUTES),
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise error

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "exp", "iat"]},
        )

        user_id = int(payload["sub"])

    except (InvalidTokenError, ValueError, TypeError, KeyError):
        raise error

    user = db.get(User, user_id)

    if user is None:
        raise error

    return user
