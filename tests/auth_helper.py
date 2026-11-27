"""
Used by the API tests: a TEMPORARY users file with two users and ready-made login headers.
Your real data/processed/users.json is never touched. Import this BEFORE backend.main.
"""

import tempfile
from pathlib import Path

from app import auth, config

config.USERS_FILE = Path(tempfile.mkdtemp()) / "users.json"
config.AUTH_SECRET = "test-secret-not-real"

PASSWORD = "Strong123"
for _name in ("test_user", "test_other"):
    auth.create_user(_name, PASSWORD)


def headers(name: str = "test_user") -> dict:
    """Authorization header for a test user."""
    token = auth.login(name, PASSWORD)["token"]
    return {"Authorization": f"Bearer {token}"}
