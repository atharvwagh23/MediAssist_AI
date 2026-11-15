"""
Login and signup endpoints (added to the main app with include_router).

    POST /auth/signup   create an account (not admin)
    POST /auth/login    returns a token
    GET  /auth/me       who am I (needs the token)
    GET  /auth/roles    roles for the signup form

Other endpoints will use `current_user` to require a login (done in step 12).
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from app import auth

router = APIRouter(prefix="/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)  # makes the "Authorize" button appear in /docs


class SignupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown fields
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=128)
    role: str = Field(min_length=1, max_length=30)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=128)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    """Use as: user: dict = Depends(current_user). Raises 401 if not logged in."""
    user = auth.verify_token(credentials.credentials if credentials else "")
    if not user:
        raise HTTPException(401, "Please log in.", headers={"WWW-Authenticate": "Bearer"})
    return user


@router.post("/signup", status_code=201)
def signup(req: SignupRequest):
    try:
        return auth.create_user(req.username, req.password, req.role)
    except auth.AuthError as error:
        raise HTTPException(error.status, error.message)


@router.post("/login")
def login(req: LoginRequest):
    try:
        return auth.login(req.username, req.password)
    except auth.AuthError as error:
        raise HTTPException(error.status, error.message)


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return user


@router.get("/roles")
def roles():
    return {"roles": auth.SIGNUP_ROLES, "tools_by_role": auth.ROLE_TOOLS}