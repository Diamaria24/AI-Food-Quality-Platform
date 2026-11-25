from fastapi import Request
from fastapi.responses import JSONResponse
from jwt import InvalidTokenError
import jwt

from backend.auth_security import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
)
from backend.database import SessionLocal
from backend.user_models import User


async def require_prediction_auth(request: Request, call_next):
    # Allow browser CORS preflight requests to reach CORSMiddleware.
    # Actual prediction requests still require a valid JWT.
    if request.method == "OPTIONS":
        return await call_next(request)

    if not request.url.path.startswith("/predict/"):
        return await call_next(request)

    authorization = request.headers.get("Authorization", "")

    if not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:].strip()

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "iat", "exp"]},
        )

        user_id = int(payload["sub"])

    except (InvalidTokenError, ValueError, TypeError, KeyError):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or expired token."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    with SessionLocal() as db:
        user = db.get(User, user_id)

        if user is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "User account not found."},
                headers={"WWW-Authenticate": "Bearer"},
            )

    request.state.user_id = user_id

    return await call_next(request)
