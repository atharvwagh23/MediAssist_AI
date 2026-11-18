"""
Login / signup tests. They use a TEMPORARY users file, so your real
data/processed/users.json is never touched.

Run from the project root:
    python -m tests.test_auth
"""

import json
import tempfile
from pathlib import Path

from app import auth, config

config.USERS_FILE = Path(tempfile.mkdtemp()) / "users.json"  # temporary file for the whole test
config.AUTH_SECRET = "test-secret-not-real"
results = []  # (test name, passed?)


def check(name: str, condition: bool) -> None:  # record and print one result
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}")


def fails_with(function, status: int) -> bool:  # True if the call raises AuthError with this status
    try:
        function()
    except auth.AuthError as error:
        return error.status == status
    return False


# ---- 1. signup rules ----
user = auth.create_user("Dr.Rao", "Strong123")
check("signup: creates a user, username is lowercased", user == {"username": "dr.rao"})
saved = json.loads(config.USERS_FILE.read_text(encoding="utf-8"))
check("signup: password is stored as a bcrypt hash, never as text",
      saved["dr.rao"]["password_hash"].startswith("$2") and "Strong123" not in config.USERS_FILE.read_text())
check("signup: no role is stored", "role" not in saved["dr.rao"])
check("signup: duplicate username gives 409", fails_with(lambda: auth.create_user("dr.rao", "Strong123"), 409))
check("signup: short password is refused", fails_with(lambda: auth.create_user("bob1", "ab1"), 400))
check("signup: password without a number is refused", fails_with(lambda: auth.create_user("bob2", "onlyletters"), 400))
check("signup: password over 72 bytes is refused", fails_with(lambda: auth.create_user("bob3", "a1" * 40), 400))
check("signup: bad username is refused", fails_with(lambda: auth.create_user("a b!", "Strong123"), 400))

# ---- 2. login ----
result = auth.login("DR.RAO", "Strong123")
check("login: correct password gives a token", "." in result["token"] and "role" not in result)
check("login: wrong password gives 401", fails_with(lambda: auth.login("dr.rao", "Wrong123"), 401))
check("login: unknown user gives the same 401", fails_with(lambda: auth.login("nobody", "Strong123"), 401))

# ---- 3. tokens ----
info = auth.verify_token(result["token"])
check("token: valid token gives the user", info == {"username": "dr.rao"})
check("token: tampered token is rejected", auth.verify_token(result["token"][:-3] + "abc") is None)
check("token: garbage is rejected", auth.verify_token("not-a-token") is None and auth.verify_token("") is None)
config.TOKEN_HOURS = -1  # a token that is already expired
expired = auth.create_token("dr.rao")
config.TOKEN_HOURS = 8
check("token: expired token is rejected", auth.verify_token(expired) is None)
config.AUTH_SECRET = "another-secret"  # a token from another server/secret
check("token: token signed with another secret is rejected", auth.verify_token(result["token"]) is None)
config.AUTH_SECRET = "test-secret-not-real"
users = json.loads(config.USERS_FILE.read_text(encoding="utf-8"))
del users["dr.rao"]
config.USERS_FILE.write_text(json.dumps(users), encoding="utf-8")
check("token: deleted user's token stops working", auth.verify_token(result["token"]) is None)

# ---- 4. lockout ----
auth.create_user("locky", "Strong123")
for _ in range(auth.MAX_FAILURES):
    fails_with(lambda: auth.login("locky", "Wrong123"), 401)
check("lockout: 5 wrong passwords lock the account (429)", fails_with(lambda: auth.login("locky", "Strong123"), 429))

# ---- 5. API (in-process) ----
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)
r = client.post("/auth/signup", json={"username": "api_user", "password": "Strong123"})
check("API: POST /auth/signup returns 201", r.status_code == 201 and r.json() == {"username": "api_user"})
r = client.post("/auth/signup", json={"username": "api_user", "password": "Strong123"})
check("API: duplicate signup returns 409", r.status_code == 409)
r = client.post("/auth/signup", json={"username": "x", "password": "Strong123", "role": "doctor"})
check("API: an extra field (such as role) returns 422", r.status_code == 422)
r = client.post("/auth/login", json={"username": "api_user", "password": "Wrong123"})
check("API: wrong password returns 401", r.status_code == 401)
r = client.post("/auth/login", json={"username": "api_user", "password": "Strong123"})
token = r.json()["token"]
check("API: login returns a token", r.status_code == 200 and token)
r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
check("API: GET /auth/me works with the token", r.status_code == 200 and r.json()["username"] == "api_user")
check("API: GET /auth/me without a token returns 401", client.get("/auth/me").status_code == 401)
check("API: GET /auth/me with a bad token returns 401",
      client.get("/auth/me", headers={"Authorization": "Bearer abc.def"}).status_code == 401)
check("API: /auth/roles no longer exists", client.get("/auth/roles").status_code == 404)
check("Login password never appears in the users file", "Strong123" not in config.USERS_FILE.read_text())

passed = sum(1 for _, ok in results if ok)
print(f"\n{passed}/{len(results)} tests passed")
raise SystemExit(0 if passed == len(results) else 1)