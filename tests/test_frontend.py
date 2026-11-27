"""
Frontend tests: the API client (against a tiny fake server) and the helper functions.
No Streamlit, no real backend, no LLM.

Run from the project root:
    python -m tests.test_frontend
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from frontend import api, helpers

results = []  # (test name, passed?)


def check(name: str, condition: bool) -> None:  # record and print one result
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}")


seen = {}  # what the fake server received


class Fake(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_POST(self):
        body = self._read()
        seen.update(path=self.path, auth=self.headers.get("Authorization"), body=body)
        if self.path == "/auth/login":
            ok = json.loads(body)["password"] == "Strong123"
            return self._send(200, {"token": "t", "username": "u"}) if ok else \
                self._send(401, {"detail": "Invalid username or password."})
        if self.path == "/chat":
            if json.loads(body)["message"] == "bad":
                return self._send(422, {"detail": [{"loc": ["body"], "msg": "technical"}]})
            return self._send(200, {"answer": "ok"})
        if self.path == "/upload":
            return self._send(200, {"source": "a.txt", "chunks": 2})
        self._send(404, {"detail": "nope"})

    def do_GET(self):
        seen.update(path=self.path, auth=self.headers.get("Authorization"))
        if self.path == "/sources":
            return self._send(200, {"sources": [{"name": "a.pdf", "chunks": 3}]})
        if self.path == "/images":
            return self._send(401, {"detail": "Please log in."})
        if self.path == "/boom":
            return self._send(500, "not an object")
        self._send(404, {"detail": "nope"})

    def do_DELETE(self):
        seen.update(path=self.path)
        self._send(200, {"deleted": self.path})

    def log_message(self, *args):  # keep the test output clean
        pass


server = HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=server.serve_forever, daemon=True).start()
api.BASE_URL = f"http://127.0.0.1:{server.server_port}"


def error_of(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except api.ApiError as error:
        return error
    return None


# ---- API client ----
check("api: login returns the token", api.login("u", "Strong123")["token"] == "t")
error = error_of(api.login, "u", "wrong")
check("api: a 401 becomes ApiError with the server's message", error.status == 401 and "Invalid" in error.message)
api.sources("my-token")
check("api: the token is sent as a Bearer header", seen["auth"] == "Bearer my-token")
check("api: sources are unwrapped to a list", api.sources("t")[0]["name"] == "a.pdf")
error = error_of(api.chat, "t", "bad")
check("api: a 422 gives a simple message, not technical details", error.status == 422 and "technical" not in error.message)
error = error_of(api.images, "t")
check("api: images 401 keeps status 401 (so the page can log out)", error.status == 401)
api.chat("t", "hello", "a" * 32, ["earlier question"])
sent = json.loads(seen["body"])
check("api: chat sends message, extraction id and history", sent == {
    "message": "hello", "history": ["earlier question"], "extraction_id": "a" * 32})
api.chat("t", "hello")
check("api: chat without an image sends no extraction_id", "extraction_id" not in json.loads(seen["body"]))
check("api: document upload works", api.upload_document("t", "a.txt", b"hello", "text/plain")["chunks"] == 2)
api.delete_image("t", "b" * 32)
check("api: delete image calls the right path", seen["path"] == "/images/" + "b" * 32)
check("api: a non-JSON error body still gives a safe message",
      error_of(api._request, "GET", "/boom").message == "Something went wrong.")
server.shutdown()
api.BASE_URL = "http://127.0.0.1:9"  # nothing listens here
error = error_of(api.sources, "t")
check("api: backend not running gives a clear message", error.status == 0 and "Is it running" in error.message)

# ---- helpers ----
turns = []
for n in range(1, 8):
    turns = helpers.push_turn(turns, {"question": f"q{n}"}, 5)
check("helpers: only the last 5 turns are kept, oldest dropped first (FIFO)", [t["question"] for t in turns] == ["q3", "q4", "q5", "q6", "q7"])
check("helpers: history for the backend is the last 3 questions", helpers.history_for_backend(turns) == ["q5", "q6", "q7"])
check("helpers: history is empty at the start", helpers.history_for_backend([]) == [])
check("helpers: trace rows are readable", helpers.trace_rows([{"agent": "mcp", "seconds": 1.2, "note": "1 ok"}])
      == [{"Agent": "mcp", "Seconds": 1.2, "What it did": "1 ok"}])
lines = helpers.source_lines([
    {"type": "document", "source": "SOP.pdf", "page": 4, "score": 0.62},
    {"type": "database", "source": "get_lab_results", "detail": "P00001"},
    {"type": "image", "source": "rx.png"}, {"source": "old.pdf", "page": 1}])
check("helpers: source lines for documents, database, images and old-style sources", lines == [
    "SOP.pdf (page 4, score 0.62)", "Database: get_lab_results (patient P00001)", "Uploaded image: rx.png", "old.pdf (page 1)"])
check("helpers: missing numbers show as n/a", helpers.show(None) == "n/a" and helpers.show(0.9) == "0.90" and helpers.show(5) == "5")
check("helpers: the typewriter gives back exactly the same text",
      "".join(helpers.typewriter("Admission starts with registration of the patient")).strip()
      == "Admission starts with registration of the patient")
check("helpers: decisions have plain-language text",
      "Blocked by the judge" == helpers.decision_text({"decision": "blocked_by_judge"}) and helpers.decision_text({}) == "n/a")

passed = sum(1 for _, ok in results if ok)
print(f"\n{passed}/{len(results)} tests passed")
raise SystemExit(0 if passed == len(results) else 1)
