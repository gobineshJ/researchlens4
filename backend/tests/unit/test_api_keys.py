"""
API keys: the X-API-Key header machine clients (n8n, scripts) use instead
of the browser's login + refresh cookie.
"""

from fastapi.testclient import TestClient

from app import db
from app.auth import create_access_token, create_api_key, hash_password
from app.main import app

client = TestClient(app)


def _session(username="alice") -> dict:
    if db.get_user(username) is None:
        db.create_user(username, hash_password("correct-horse"))
    return {"Authorization": f"Bearer {create_access_token(username)}"}


def _new_key(headers, name="n8n") -> dict:
    response = client.post("/auth/api-keys", json={"name": name}, headers=headers)
    assert response.status_code == 201
    return response.json()


def test_key_authenticates_protected_routes():
    key = _new_key(_session())
    assert key["api_key"].startswith("rl_")

    me = client.get("/auth/me", headers={"X-API-Key": key["api_key"]})
    assert me.status_code == 200
    assert me.json() == {"username": "alice"}
    assert client.get("/evaluation-history", headers={"X-API-Key": key["api_key"]}).status_code == 200


def test_key_is_stored_hashed_and_never_listed():
    headers = _session()
    key = _new_key(headers)
    with db.connect() as conn:
        stored = conn.execute("SELECT key_hash FROM api_keys").fetchone()["key_hash"]
    assert stored != key["api_key"] and key["api_key"] not in stored

    listed = client.get("/auth/api-keys", headers=headers).json()
    assert [k["key_id"] for k in listed] == [key["key_id"]]
    assert "api_key" not in listed[0]
    assert key["api_key"].startswith(listed[0]["prefix"])


def test_use_is_recorded():
    headers = _session()
    key = _new_key(headers)
    client.get("/auth/me", headers={"X-API-Key": key["api_key"]})
    assert client.get("/auth/api-keys", headers=headers).json()[0]["last_used_at"] is not None


def test_garbage_key_is_rejected():
    response = client.get("/auth/me", headers={"X-API-Key": "rl_not-a-real-key"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_revoked_key_stops_working():
    headers = _session()
    key = _new_key(headers)
    assert client.delete(f"/auth/api-keys/{key['key_id']}", headers=headers).status_code == 204
    assert client.get("/auth/me", headers={"X-API-Key": key["api_key"]}).status_code == 401
    assert client.get("/auth/api-keys", headers=headers).json()[0]["revoked"] is True


def test_cannot_revoke_someone_elses_key():
    alice_key = _new_key(_session("alice"))
    response = client.delete(f"/auth/api-keys/{alice_key['key_id']}", headers=_session("bob"))
    assert response.status_code == 404
    assert client.get("/auth/me", headers={"X-API-Key": alice_key["api_key"]}).status_code == 200


def test_key_dies_with_its_account():
    _session("alice")
    key = create_api_key("alice", "n8n")
    with db.connect() as conn:
        conn.execute("DELETE FROM users WHERE username = 'alice'")
    assert client.get("/auth/me", headers={"X-API-Key": key.api_key}).status_code == 401


def test_a_key_cannot_manage_keys():
    # A leaked key must not be enough to mint more keys or list/revoke the
    # owner's others -- that takes a real login.
    key = _new_key(_session())
    as_key = {"X-API-Key": key["api_key"]}
    assert client.post("/auth/api-keys", json={"name": "more"}, headers=as_key).status_code == 401
    assert client.get("/auth/api-keys", headers=as_key).status_code == 401
    assert client.delete(f"/auth/api-keys/{key['key_id']}", headers=as_key).status_code == 401


def test_key_name_is_validated():
    headers = _session()
    assert client.post("/auth/api-keys", json={"name": ""}, headers=headers).status_code == 422
    assert client.post("/auth/api-keys", json={"name": "x" * 65}, headers=headers).status_code == 422
