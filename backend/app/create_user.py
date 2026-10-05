"""
Create an account from the command line -- the way to add users once
ALLOW_REGISTRATION is turned off.

    python -m app.create_user alice
"""

import getpass
import sys

from . import db
from .auth import RegisterRequest, hash_password


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m app.create_user <username>")
        return 2
    username = sys.argv[1]
    password = getpass.getpass("Password (min 8 characters): ")
    try:
        RegisterRequest(username=username, password=password)
    except ValueError as exc:
        print(exc)
        return 1
    if not db.create_user(username, hash_password(password)):
        print(f"User {username!r} already exists.")
        return 1
    print(f"Created user {username!r}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
