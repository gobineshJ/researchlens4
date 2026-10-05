"""
OAuth2 password flow with JWT bearer tokens.

  POST /auth/token     form fields username + password (the OAuth2 spec's
                       format, which is also what Swagger's "Authorize"
                       button sends) -> {"access_token", "token_type"}
  POST /auth/register  JSON {username, password} -> creates an account
                       (only while ALLOW_REGISTRATION is on)
  GET  /auth/me        who the current token belongs to

Every other route depends on get_current_user, which rejects any request
without a valid "Authorization: Bearer <token>" header with a 401.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from . import config, db

logger = logging.getLogger(__name__)

if config.JWT_SECRET_KEY:
    _SECRET_KEY = config.JWT_SECRET_KEY
else:
    _SECRET_KEY = secrets.token_urlsafe(48)
    logger.warning(
        "JWT_SECRET_KEY is not set -- using a random key, so every token "
        "becomes invalid when the API restarts. Set it in backend/.env."
    )

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(min_length=8, max_length=72)
    # bcrypt only looks at the first 72 bytes of a password; rejecting
    # longer ones is clearer than silently ignoring the rest.


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class User(BaseModel):
    username: str


# --- passwords ---

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


# Checked against when the username doesn't exist, so a wrong username
# takes as long as a wrong password and response timing can't be used to
# discover which usernames are registered.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def authenticate(username: str, password: str) -> User | None:
    row = db.get_user(username)
    if row is None:
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return User(username=row["username"])


# --- tokens ---

def create_access_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        raise unauthorized
    username = payload.get("sub")
    # Re-check the account still exists, so deleting a user cuts off
    # their tokens immediately rather than when the tokens expire.
    if not username or db.get_user(username) is None:
        raise unauthorized
    return User(username=username)


# --- routes ---

@router.post("/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(user.username))


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest):
    if not config.ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="Registration is disabled.")
    if not db.create_user(body.username, hash_password(body.password)):
        raise HTTPException(status_code=409, detail="That username is already taken.")
    return User(username=body.username)


@router.get("/me", response_model=User)
def me(user: User = Depends(get_current_user)):
    return user
