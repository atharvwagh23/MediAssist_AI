"""
Login and signup logic (no web code here, so it is easy to test).

* Passwords are stored ONLY as bcrypt hashes in data/processed/users.json.
* Login gives a signed token (HMAC) that expires after TOKEN_HOURS.
* 5 wrong passwords lock that username for 5 minutes.
* Roles decide which database tools a user may use (ROLE_TOOLS).

Create the first admin from the terminal:
    python -m app.auth create-admin <username>
"""

import base64
import getpass
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone

import bcrypt  # password hashing

from app import config

logger = logging.getLogger("mediassist.auth")

# What each role may use (names match the MCP tools). Change this table to change access.
ROLE_TOOLS = {
    "doctor": ["search_patients", "get_patient_history", "get_appointments", "get_lab_results"],
    "admin": ["search_patients", "get_patient_history", "get_appointments", "get_lab_results",
              "get_billing_summary"],
    "front_desk": ["search_patients", "get_appointments"],
    "insurance": ["search_patients", "get_billing_summary"],
    "clinical_ops": ["get_patient_history", "get_appointments", "get_lab_results"],
}
SIGNUP_ROLES = [role for role in ROLE_TOOLS if role != "admin"]  # nobody can sign up as admin

USERNAME_PATTERN = re.compile(r"^[a-z0-9_.]{3,30}$")
MAX_FAILURES, LOCK_SECONDS = 5, 300  # lock after 5 wrong passwords for 5 minutes

_file_lock = threading.Lock()  # one write to users.json at a time
_failures: dict[str, list[float]] = {}  # username -> times of recent wrong passwords
_fallback_secret = secrets.token_hex(32)  # used only if AUTH_SECRET is missing in .env
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-1", bcrypt.gensalt()).decode()  # makes unknown users take equal time


class AuthError(Exception):
    """A clear, safe message for the user plus the HTTP status to use."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


# ---------- users.json ----------

def _load_users() -> dict:
    path = config.USERS_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.error("users file is unreadable")
        raise AuthError("User storage is not available. Please contact the administrator.", 500)


def _save_users(users: dict) -> None:
    path = config.USERS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(users, indent=2), encoding="utf-8")  # write a temp file first...
    os.replace(temp, path)  # ...then swap it in, so a crash never leaves a half-written file


# ---------- checks ----------

def clean_username(username: str) -> str:
    username = (username or "").strip().lower()
    if not USERNAME_PATTERN.match(username):
        raise AuthError("Username must be 3 to 30 characters: letters, numbers, dot or underscore.")
    return username


def check_password_rules(password: str) -> None:
    password = password or ""
    if len(password.encode("utf-8")) > 72:  # bcrypt cannot use more than 72 bytes
        raise AuthError("Password is too long (maximum 72 characters).")
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise AuthError("Password needs at least 8 characters, with letters and numbers.")


# ---------- signup / login ----------

def create_user(username: str, password: str, role: str, allow_admin: bool = False) -> dict:
    username = clean_username(username)
    check_password_rules(password)
    role = (role or "").strip().lower().replace(" ", "_")
    allowed = list(ROLE_TOOLS) if allow_admin else SIGNUP_ROLES
    if role not in allowed:
        raise AuthError("Role must be one of: " + ", ".join(SIGNUP_ROLES) + ".")
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
    with _file_lock:
        users = _load_users()
        if username in users:
            raise AuthError("This username is already taken.", 409)
        users[username] = {"password_hash": hashed, "role": role,
                           "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _save_users(users)
    logger.info("user created role=%s", role)
    return {"username": username, "role": role}


def login(username: str, password: str) -> dict:
    username = (username or "").strip().lower()
    now = time.time()
    recent = [t for t in _failures.get(username, []) if now - t < LOCK_SECONDS]
    _failures[username] = recent
    if len(recent) >= MAX_FAILURES:
        raise AuthError("Too many wrong attempts. Please try again in 5 minutes.", 429)

    user = _load_users().get(username)
    stored = user["password_hash"] if user else _DUMMY_HASH  # same work whether or not the user exists
    try:
        correct = bcrypt.checkpw((password or "").encode("utf-8")[:72], stored.encode())
    except ValueError:
        correct = False
    if not user or not correct:
        _failures[username].append(now)
        raise AuthError("Invalid username or password.", 401)  # same message for both cases

    _failures.pop(username, None)
    return {"token": create_token(username), "username": username, "role": user["role"],
            "expires_in_hours": config.TOKEN_HOURS}


# ---------- tokens ----------

def _secret() -> bytes:
    if config.AUTH_SECRET:
        return config.AUTH_SECRET.encode()
    logger.warning("AUTH_SECRET is not set in .env; logins stop working when the server restarts.")
    return _fallback_secret.encode()


def _sign(text: str) -> str:
    return hmac.new(_secret(), text.encode(), hashlib.sha256).hexdigest()


def create_token(username: str) -> str:
    data = {"u": username, "exp": int(time.time() + config.TOKEN_HOURS * 3600)}
    body = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()
    return f"{body}.{_sign(body)}"


def verify_token(token: str) -> dict | None:
    """Return {"username", "role", "tools"} for a valid token, else None."""
    try:
        body, signature = (token or "").split(".", 1)
        if not hmac.compare_digest(signature, _sign(body)):  # tampered or from another server
            return None
        data = json.loads(base64.urlsafe_b64decode(body.encode()))
        if data["exp"] < time.time():  # expired
            return None
        user = _load_users().get(data["u"])  # role is read from the file, never from the token
    except (ValueError, KeyError, TypeError, AuthError):
        return None
    if not user:  # user was deleted
        return None
    return {"username": data["u"], "role": user["role"], "tools": ROLE_TOOLS.get(user["role"], [])}


def can_use_tool(role: str, tool_name: str) -> bool:
    return tool_name in ROLE_TOOLS.get(role, [])


# ---------- terminal helper: python -m app.auth create-admin <username> ----------

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "create-admin":
        print("Usage: python -m app.auth create-admin <username>")
        raise SystemExit(1)
    first = getpass.getpass("Password: ")  # typed text is hidden
    if first != getpass.getpass("Repeat password: "):
        print("Passwords do not match.")
        raise SystemExit(1)
    try:
        print("Created:", create_user(sys.argv[2], first, "admin", allow_admin=True))
    except AuthError as error:
        print("Error:", error.message)
        raise SystemExit(1)