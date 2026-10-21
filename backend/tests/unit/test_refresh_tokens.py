"""
Refresh tokens: an httpOnly cookie that silently renews the short-lived
access token, rotates on every use, and ends the whole login if an
already-used token is ever presented again (it was copied).
"""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app import auth, config, db
from app.main import app

COOKIE = config.REFRESH_COOKIE_NAME


def _logged_in_client(username="alice"):
    client = TestClient(app)
    client.post("/auth/register", json={"username": username, "password": "correct-horse"})
    response = client.post("/auth/token", data={"username": username, "password": "correct-horse"})
    assert response.status_code == 200
    return client, response


def _me(client, access_token):
    return client.get("/auth/me", headers={"Authorization": f"Bearer {access_token}"})


def test_login_sets_an_httponly_strict_refresh_cookie():
    client, response = _logged_in_client()
    set_cookie = response.headers["set-cookie"]
    assert set_cookie.startswith(f"{COOKIE}=")
    assert "HttpOnly" in set_cookie and "SameSite=strict" in set_cookie
    # Stored hashed, never as the token itself.
    token = client.cookies[COOKIE]
    assert db.get_refresh_token(token) is None
    assert db.get_refresh_token(auth._hash_secret(token)) is not None


def test_refresh_returns_a_working_access_token_and_rotates_the_cookie():
    client, _ = _logged_in_client()
    first = client.cookies[COOKIE]
    response = client.post("/auth/refresh")
    assert response.status_code == 200
    assert _me(client, response.json()["access_token"]).json() == {"username": "alice"}
    assert client.cookies[COOKIE] != first


def test_reusing_a_rotated_token_ends_the_whole_login():
    client, _ = _logged_in_client()
    stolen = client.cookies[COOKIE]
    assert client.post("/auth/refresh").status_code == 200   # the real user rotates
    current = client.cookies[COOKIE]

    attacker = TestClient(app)
    attacker.cookies.set(COOKIE, stolen)
    assert attacker.post("/auth/refresh").status_code == 401  # reuse detected...

    client.cookies.set(COOKIE, current)
    assert client.post("/auth/refresh").status_code == 401    # ...so the real session ends too


def test_logout_revokes_and_clears_the_cookie():
    client, _ = _logged_in_client()
    token = client.cookies[COOKIE]
    response = client.post("/auth/logout")
    assert response.status_code == 204
    assert f'{COOKIE}=""' in response.headers["set-cookie"] or "Max-Age=0" in response.headers["set-cookie"]
    replay = TestClient(app)
    replay.cookies.set(COOKIE, token)
    assert replay.post("/auth/refresh").status_code == 401


def test_refresh_without_cookie_or_with_garbage_is_rejected():
    assert TestClient(app).post("/auth/refresh").status_code == 401
    client = TestClient(app)
    client.cookies.set(COOKIE, "made-up")
    assert client.post("/auth/refresh").status_code == 401


def test_expired_refresh_token_is_rejected():
    client, _ = _logged_in_client()
    with db.connect() as conn:
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        conn.execute("UPDATE refresh_tokens SET expires_at = ?", (past,))
    assert client.post("/auth/refresh").status_code == 401


def test_refresh_for_a_deleted_user_is_rejected():
    client, _ = _logged_in_client()
    with db.connect() as conn:
        conn.execute("DELETE FROM users WHERE username = 'alice'")
    assert client.post("/auth/refresh").status_code == 401


def test_cors_admits_only_configured_origins():
    client = TestClient(app)
    allowed = client.options("/auth/me", headers={"Origin": config.CORS_ORIGINS[0],
                                                  "Access-Control-Request-Method": "GET"})
    assert allowed.headers.get("access-control-allow-origin") == config.CORS_ORIGINS[0]
    assert allowed.headers.get("access-control-allow-credentials") == "true"
    other = client.options("/auth/me", headers={"Origin": "https://evil.example",
                                                "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in other.headers
