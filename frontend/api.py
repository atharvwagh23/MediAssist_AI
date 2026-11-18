"""
Talks to the FastAPI backend. No Streamlit code here, so it is easy to test.

Every function raises ApiError with a short, safe message that can be shown to the user.
"""

import requests

from app import config

BASE_URL = config.BACKEND_URL  # tests change this to point at a fake server


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _request(method: str, path: str, token: str | None = None, timeout: int = 30, **kwargs) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.request(method, BASE_URL + path, headers=headers, timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise ApiError(0, f"Cannot reach the backend at {BASE_URL}. Is it running? "
                          "(uvicorn backend.main:app --port 8000)")
    except requests.Timeout:
        raise ApiError(0, "The server took too long to answer. Please try again.")
    if response.status_code >= 400:
        try:
            body = response.json()
            detail = body.get("detail") if isinstance(body, dict) else None
        except ValueError:
            detail = None
        if not isinstance(detail, str):  # 422 gives a list of technical details: show something simple
            detail = "The request was not valid." if response.status_code == 422 else "Something went wrong."
        raise ApiError(response.status_code, detail)
    return response.json()


def login(username: str, password: str) -> dict:
    return _request("POST", "/auth/login", json={"username": username, "password": password})


def signup(username: str, password: str) -> dict:
    return _request("POST", "/auth/signup", json={"username": username, "password": password})



def sources(token: str) -> list[dict]:
    return _request("GET", "/sources", token)["sources"]


def delete_source(token: str, name: str) -> dict:
    return _request("DELETE", f"/sources/{name}", token)


def upload_document(token: str, name: str, content: bytes, mime: str) -> dict:
    return _request("POST", "/upload", token, timeout=300, files={"file": (name, content, mime)})


def upload_image(token: str, name: str, content: bytes, mime: str) -> dict:
    return _request("POST", "/upload-image", token, timeout=300, files={"file": (name, content, mime)})


def images(token: str) -> list[dict]:
    return _request("GET", "/images", token)["images"]


def delete_image(token: str, extraction_id: str) -> dict:
    return _request("DELETE", f"/images/{extraction_id}", token)


def chat(token: str, message: str, extraction_id: str | None = None, history: list[str] | None = None) -> dict:
    body = {"message": message, "history": history or []}
    if extraction_id:
        body["extraction_id"] = extraction_id
    return _request("POST", "/chat", token, timeout=180, json=body)
