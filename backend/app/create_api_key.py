"""
Create an API key from the command line -- for setting up a machine client
(n8n, a script, an MCP server) without logging in first.

    python -m app.create_api_key alice n8n
    docker compose exec api python -m app.create_api_key alice n8n

The key is printed once; only its hash is stored. Revoke it with
DELETE /auth/api-keys/{key_id} from a logged-in session.
"""

import sys

from . import db
from .auth import ApiKeyCreateRequest, create_api_key


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.create_api_key <username> <key name>")
        return 2
    username, name = sys.argv[1], sys.argv[2]
    try:
        ApiKeyCreateRequest(name=name)
    except ValueError as exc:
        print(exc)
        return 1
    if db.get_user(username) is None:
        print(f"No user {username!r} -- create one first with: python -m app.create_user {username}")
        return 1
    key = create_api_key(username, name)
    print(f"Created API key {key.key_id!r} ({name}) for {username!r}.")
    print("Send it as the X-API-Key header. It won't be shown again:\n")
    print(key.api_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
