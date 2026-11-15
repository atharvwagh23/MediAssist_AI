"""
Used by the API tests: a TEMPORARY users file with one user per role and ready-made login headers.
Your real data/processed/users.json is never touched. Import this BEFORE backend.main.
"""

import tempfile
from pathlib import Path

from app import auth, config

config.USERS_FILE = Path(tempfile.mkdtemp()) / "users.json"
config.AUTH_SECRET = "test-secret-not-real"

PASSWORD = "Strong123"
for _role in ("doctor", "front_desk"):
    auth.create_user(f"test_{_role}", PASSWORD, _role)
auth.create_user("test_admin", PASSWORD, "admin", allow_admin=True)


def headers(role: str) -> dict:
    """Authorization header for a test user of this role."""
    token = auth.login(f"test_{role}", PASSWORD)["token"]
    return {"Authorization": f"Bearer {token}"}
