"""
OAuth2 password flow with JWT bearer tokens.

  POST /auth/token     form fields username + password (the OAuth2 spec's
                       format, which is also what Swagger's "Authorize"
                       button sends) -> {"access_token", "token_type"},
                       plus a refresh token in an httpOnly cookie
  POST /auth/refresh   refresh cookie -> a new access token (and a new,
                       rotated refresh cookie)
  POST /auth/logout    revokes the refresh cookie's login and clears it
  POST /auth/register  JSON {username, password} -> creates an account
                       (only while ALLOW_REGISTRATION is on)
  GET  /auth/me        who the current token belongs to
  POST   /auth/api-keys            {name} -> a new API key, shown only this once
  GET    /auth/api-keys            your keys (never the secret)
  DELETE /auth/api-keys/{key_id}   revokes one

Access tokens are short-lived JWTs the browser keeps only in memory.
Refresh tokens are random strings, stored hashed, that rotate on every
use. If a refresh token is presented a second time -- which only happens
when it was stolen and used by someone else first, or vice versa -- the
whole login ("family") is revoked, logging out both parties.

Every other route depends on get_current_user, which rejects any request
without a valid "Authorization: Bearer <token>" header -- or a valid
"X-API-Key: rl_..." header -- with a 401.

API keys are for machine clients (n8n, scripts, an MCP server) that can't
log in and refresh like a browser. They are random strings stored hashed,
never expire, and stop working the moment they're revoked or their account
is deleted. Managing keys needs a real login (get_session_user), so a leaked
key can't be used to mint more keys or to revoke the owner's other ones.
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer, OAuth2PasswordRequestForm
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

# auto_error=False on both: a request may carry either one, so a missing
# header alone mustn't reject it -- get_current_user decides.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)
api_key_scheme = APIKeyHeader(name=config.API_KEY_HEADER_NAME, auto_error=False)

API_KEY_PREFIX = "rl_"

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


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class ApiKeyInfo(BaseModel):
    key_id: str
    name: str
    prefix: str
    revoked: bool
    last_used_at: str | None
    created_at: str


class ApiKeyCreated(ApiKeyInfo):
    api_key: str  # the full key -- returned once, at creation, and never again


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


def _unauthorized(detail: str = "Invalid or expired token.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _user_from_token(token: str) -> User:
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        raise _unauthorized()
    username = payload.get("sub")
    # Re-check the account still exists, so deleting a user cuts off
    # their tokens immediately rather than when the tokens expire.
    if not username or db.get_user(username) is None:
        raise _unauthorized()
    return User(username=username)


def _user_from_api_key(api_key: str) -> User:
    row = db.get_api_key_by_hash(_hash_secret(api_key))
    # Same check as for tokens: a deleted account's keys die with it.
    if row is None or row["revoked"] or db.get_user(row["username"]) is None:
        raise _unauthorized("Invalid or revoked API key.")
    db.touch_api_key(row["key_id"], datetime.now(timezone.utc).isoformat())
    return User(username=row["username"])


def get_current_user(token: str | None = Depends(oauth2_scheme),
                     api_key: str | None = Depends(api_key_scheme)) -> User:
    """A logged-in user (Bearer token) or a machine client (X-API-Key)."""
    if api_key:
        return _user_from_api_key(api_key)
    if token:
        return _user_from_token(token)
    raise _unauthorized("Not authenticated.")


def get_session_user(token: str | None = Depends(oauth2_scheme)) -> User:
    """A logged-in user only -- API keys are not accepted."""
    if not token:
        raise _unauthorized("This needs a login session (Bearer token), not an API key.")
    return _user_from_token(token)


# --- API keys ---

def create_api_key(username: str, name: str) -> ApiKeyCreated:
    api_key = API_KEY_PREFIX + secrets.token_urlsafe(32)
    key_id = uuid.uuid4().hex[:12]
    prefix = api_key[:len(API_KEY_PREFIX) + 6]
    db.store_api_key(key_id, _hash_secret(api_key), prefix, username, name)
    return ApiKeyCreated(key_id=key_id, name=name, prefix=prefix, revoked=False,
                         last_used_at=None, created_at=datetime.now(timezone.utc).isoformat(),
                         api_key=api_key)


# --- refresh tokens ---

def _hash_secret(secret: str) -> str:
    # A plain (unsalted) hash is enough for refresh tokens and API keys:
    # unlike passwords they are 32 random bytes, far beyond brute force.
    return hashlib.sha256(secret.encode()).hexdigest()


def issue_refresh_token(username: str, family_id: str | None = None) -> str:
    """A new random refresh token, stored hashed. `family_id` ties a rotated
    token to the login it descends from."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    db.delete_expired_refresh_tokens(now.isoformat())
    db.store_refresh_token(
        _hash_secret(token), username, family_id or uuid.uuid4().hex,
        (now + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)).isoformat(),
    )
    return token


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        config.REFRESH_COOKIE_NAME, token,
        max_age=config.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        httponly=True,             # unreadable from JavaScript
        secure=config.COOKIE_SECURE,
        samesite="strict",         # never sent by another site's requests (no CSRF)
        path="/",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(config.REFRESH_COOKIE_NAME, path="/", httponly=True,
                           secure=config.COOKIE_SECURE, samesite="strict")


# --- routes ---

@router.post("/token", response_model=Token)
def login(response: Response, form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _set_refresh_cookie(response, issue_refresh_token(user.username))
    return Token(access_token=create_access_token(user.username))


@router.post("/refresh", response_model=Token)
def refresh(response: Response, refresh_token: str | None = Cookie(default=None, alias=config.REFRESH_COOKIE_NAME)):
    def reject(detail: str):
        # The cookie is useless from here on either way -- clear it, so the
        # browser stops sending it.
        error = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
        error.headers = {"set-cookie": f"{config.REFRESH_COOKIE_NAME}=; Max-Age=0; Path=/; HttpOnly; SameSite=strict"}
        return error

    if not refresh_token:
        raise reject("Not logged in.")
    token_hash = _hash_secret(refresh_token)
    row = db.get_refresh_token(token_hash)
    if row is None:
        raise reject("Session not recognised -- please log in again.")
    if row["revoked"] or not db.revoke_refresh_token(token_hash):
        # Reuse of a rotated token: someone else has (or had) a copy.
        # End the whole login so the stolen copy stops working too.
        logger.warning("Refresh token reuse for user %s -- revoking that login", row["username"])
        db.revoke_refresh_family(row["family_id"])
        raise reject("Session ended -- please log in again.")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        raise reject("Session expired -- please log in again.")
    if db.get_user(row["username"]) is None:
        raise reject("Account no longer exists.")

    _set_refresh_cookie(response, issue_refresh_token(row["username"], row["family_id"]))
    return Token(access_token=create_access_token(row["username"]))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, refresh_token: str | None = Cookie(default=None, alias=config.REFRESH_COOKIE_NAME)):
    if refresh_token:
        row = db.get_refresh_token(_hash_secret(refresh_token))
        if row is not None:
            db.revoke_refresh_family(row["family_id"])
    _clear_refresh_cookie(response)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


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


@router.post("/api-keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
def new_api_key(body: ApiKeyCreateRequest, user: User = Depends(get_session_user)):
    return create_api_key(user.username, body.name)


@router.get("/api-keys", response_model=list[ApiKeyInfo])
def list_api_keys(user: User = Depends(get_session_user)):
    return [ApiKeyInfo(**dict(row)) for row in db.list_api_keys(user.username)]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(key_id: str, user: User = Depends(get_session_user)):
    if not db.revoke_api_key(user.username, key_id):
        raise HTTPException(status_code=404, detail="API key not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
