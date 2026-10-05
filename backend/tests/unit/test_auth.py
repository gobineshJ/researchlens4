"""
Login, registration and token validation -- the OAuth2 password flow the
login screen uses.
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from app import auth, config
from app.main import app

client = TestClient(app)


def _register(username="alice", password="correct-horse"):
    return client.post("/auth/register", json={"username": username, "password": password})


def _login(username="alice", password="correct-horse"):
    return client.post("/auth/token", data={"username": username, "password": password})


def test_register_then_login_then_me():
    assert _register().status_code == 201

    response = _login()
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json() == {"username": "alice"}


def test_password_is_stored_hashed():
    from app import db
    _register()
    stored = db.get_user("alice")["password_hash"]
    assert stored != "correct-horse"
    assert auth.verify_password("correct-horse", stored)


def test_register_rejects_duplicate_username():
    _register()
    assert _register().status_code == 409


@pytest.mark.parametrize("username,password", [
    ("al", "correct-horse"),            # username too short
    ("bad name", "correct-horse"),      # space not allowed
    ("alice", "short"),                 # password too short
    ("alice", "x" * 73),                # past bcrypt's 72-byte limit
])
def test_register_validates_input(username, password):
    assert _register(username, password).status_code == 422


def test_register_disabled(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_REGISTRATION", False)
    assert _register().status_code == 403


def test_login_wrong_password():
    _register()
    response = _login(password="wrong-password")
    assert response.status_code == 401
    assert "access_token" not in response.json()


def test_login_unknown_user():
    assert _login(username="nobody").status_code == 401


def test_expired_token_rejected():
    _register()
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    token = jwt.encode({"sub": "alice", "iat": past, "exp": past + timedelta(minutes=1)},
                       auth._SECRET_KEY, algorithm=config.JWT_ALGORITHM)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_signed_with_other_key_rejected():
    _register()
    token = jwt.encode({"sub": "alice", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
                       "some-other-secret-that-is-at-least-32-bytes", algorithm=config.JWT_ALGORITHM)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_for_deleted_user_rejected():
    # A valid signature isn't enough: the account has to still exist.
    token = auth.create_access_token("ghost")
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
