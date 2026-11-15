"""
API test: runs every endpoint in-process (no server needed).

Run from the project root (after python -m app.rag.ingest):
    python -m tests.test_api
"""

from fastapi.testclient import TestClient  # calls the API without starting uvicorn

from tests import auth_helper  # temporary users (import before backend.main)
from backend.main import app  # noqa: E402

client = TestClient(app, headers=auth_helper.headers("admin"))  # every call is logged in as admin
anonymous = TestClient(app)  # no token
results = []  # (test name, passed?)


def check(name: str, condition: bool) -> None:  # record and print one result
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}")


# 1. health
r = client.get("/health")
check("GET /health returns ok and indexed chunks", r.status_code == 200 and r.json()["indexed_chunks"] > 0)

check("Endpoints need a login (401 without a token)",
      anonymous.post("/retrieve", json={"query": "refund"}).status_code == 401
      and anonymous.post("/chat", json={"message": "hello"}).status_code == 401)

# 2. retrieve
r = client.post("/retrieve", json={"query": "What is the admission process?"})
data = r.json()
check("POST /retrieve finds chunks for a hospital question", r.status_code == 200 and data["found"])

r = client.post("/retrieve", json={"query": "What is the capital of France?"})
check("POST /retrieve finds nothing for a non-medical question", r.json()["found"] is False)

r = client.post("/retrieve", json={"query": ""})
check("POST /retrieve rejects an empty query (422)", r.status_code == 422)

# 3. chat
r = client.post("/chat", json={"message": "What is the admission process?"})
chat = r.json()
check("POST /chat answers with sources", r.status_code == 200 and chat["found"] and len(chat["sources"]) > 0)
print("      answer preview:", chat["answer"][:150].replace("\n", " "))

r = client.post("/chat", json={"message": "What is the capital of France?"})
check("POST /chat refuses to guess when nothing is found", r.json()["found"] is False)

# 4. upload -> retrieve inside that file -> delete
text = b"Refund policy test: patient refunds are processed within seven working days after approval."
r = client.post("/upload", files={"file": ("zz_test_refund.txt", text, "text/plain")})
check("POST /upload indexes a TXT file", r.status_code == 200 and r.json()["chunks"] >= 1)

r = client.post("/retrieve", json={"query": "How many days does a refund take?", "source": "zz_test_refund.txt"})
check("Uploaded file is searchable", r.json()["found"])

r = client.post("/upload", files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
check("POST /upload rejects .exe (400)", r.status_code == 400)

r = client.post("/upload", files={"file": ("empty.txt", b"", "text/plain")})
check("POST /upload rejects an empty file (400)", r.status_code == 400)

r = client.get("/sources")
check("GET /sources lists the uploaded file", "zz_test_refund.txt" in [s["name"] for s in r.json()["sources"]])

r = client.delete("/sources/zz_test_refund.txt")
check("DELETE /sources removes the file", r.status_code == 200)

r = client.delete("/sources/zz_test_refund.txt")
check("DELETE again returns 404", r.status_code == 404)

passed = sum(1 for _, ok in results if ok)
print(f"\n{passed}/{len(results)} tests passed")