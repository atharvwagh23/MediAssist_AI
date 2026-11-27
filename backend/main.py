"""
FastAPI backend for MediAssist AI.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000/docs to try every endpoint.

Login is required for everything except /health and /auth/*.
Click "Authorize" in /docs and paste the token from POST /auth/login.

Endpoints:
    GET    /health         is the API alive, what is indexed
    POST   /auth/...       signup, login, me (see backend/auth_routes.py)
    POST   /upload         upload a document (indexed) or an image (OCR + analysis)
    POST   /upload-image   upload a png/jpg/jpeg only: OCR + vision + extraction
    POST   /retrieve       semantic search, returns chunks with scores
    POST   /chat           guardrails -> six agents (LangGraph) -> answer with sources and evaluation
    GET    /images         the user's analysed images
    DELETE /images/{id}    remove an analysed image
    GET    /sources        list indexed files
    DELETE /sources/{name} remove a file from the knowledge base 
"""

import json
import logging  # server-side error logs
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4  # random file names for images

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool  # runs slow work (OCR) without freezing the API
from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.agents.graph import run_graph
from app.guardrails.file_guard import check_document_text, check_file
from app.guardrails.input_guard import check_input
from app.guardrails.pii import mask_for_log
from app.multimodal.report_analyzer import analyze_image, get_extraction
from app.rag.loader import load_document
from app.rag.retriever import retrieve
from app.rag.vector_store import add_file, list_sources, remove_source
from backend.auth_routes import current_user
from backend.auth_routes import router as auth_router  # /auth/signup, /auth/login, /auth/me

logger = logging.getLogger("mediassist")


def _warm_up() -> None:
    """Load the slow models once, in the background, so the first user is not kept waiting."""
    try:
        from app.rag.embedder import embed_texts
        embed_texts(["warm up"])  # loads the embedding model
        from app.multimodal import ocr
        ocr.get_reader()  # loads EasyOCR (about 8 seconds)
        logger.info("Models are warm.")
    except Exception:
        logger.exception("Warm-up failed (the models will load on first use instead)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("WARM_UP", "true").lower() != "false":  # set WARM_UP=false on small servers (saves RAM)
        threading.Thread(target=_warm_up, daemon=True).start()  # does not block the server start
    yield


app = FastAPI(title="MediAssist AI API", version="0.2.0", lifespan=lifespan)
app.include_router(auth_router)  # adds the login endpoints


# ---------- request shapes (FastAPI validates these automatically) ----------

class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown fields
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    source: str | None = None  # optional: search inside one file only


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    extraction_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")  # id from /upload-image
    history: list[str] = Field(default_factory=list, max_length=5)  # last user messages (for follow-ups)


# ---------- helpers ----------

async def _handle_image(check, content: bytes, user: dict) -> dict:
    """Save the image, run OCR + vision + extraction, return the result."""
    saved = config.IMAGES_DIR / f"{uuid4().hex}.{check.ext}"  # random name, never the user's name
    saved.write_bytes(content)
    try:
        result = await run_in_threadpool(analyze_image, saved)  # heavy work runs in a worker thread
    except ValueError as exc:  # our own clear messages (damaged image, hidden instructions)
        saved.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not process '{check.safe_name}': {exc}")
    except Exception:  # never show raw errors (they can contain keys or paths)
        logger.exception("Image analysis failed for %s", check.safe_name)
        saved.unlink(missing_ok=True)
        raise HTTPException(500, f"Image analysis failed for '{check.safe_name}'. Check the server log.")
    result["original_name"] = check.safe_name  # shown in the image list
    result["owner"] = user["username"]  # only this user can use or see it
    (config.EXTRACTIONS_DIR / f"{result['id']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"kind": "image", **result}


EXTRACTION_ID = re.compile(r"^[0-9a-f]{32}$")


def _visible(data: dict, user: dict) -> bool:
    """An image result belongs to the user who uploaded it. Old results with no owner are open."""
    return data.get("owner") in (None, user["username"])


# ---------- endpoints ----------

@app.get("/health")
def health():
    sources = list_sources()
    return {"status": "ok", "indexed_chunks": sum(sources.values()), "sources": list(sources)}


@app.post("/upload")
async def upload(file: UploadFile = File(...), user: dict = Depends(current_user)):
    content = await file.read()
    check = check_file(file.filename, content)  # type, size, real file type, macros, scripts...
    if not check.ok:
        raise HTTPException(check.status, check.message)

    if check.kind == "image":  # images go to OCR + vision + extraction (any logged-in user)
        return await _handle_image(check, content, user)

    safe_name = check.safe_name
    path = config.RAW_DIR / safe_name
    existed = path.exists()
    path.write_bytes(content)
    try:
        text = "\n".join(p["text"] for p in load_document(path))  # read it once to scan the text
        text_check = check_document_text(text)  # prompt injection hidden inside the document
        if not text_check.ok:
            raise ValueError(text_check.message)
        info = add_file(path)  # load -> chunk -> embed -> index
    except ValueError as exc:  # our own clear messages (e.g. no readable text)
        if not existed:
            path.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not process '{safe_name}': {exc}")
    except Exception:
        logger.exception("Ingestion failed for %s", safe_name)
        if not existed:
            path.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed for '{safe_name}'. Check the server log.")
    return {"kind": "document", **info}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), user: dict = Depends(current_user)):
    content = await file.read()
    check = check_file(file.filename, content)  # same safety checks as /upload
    if not check.ok:
        raise HTTPException(check.status, check.message)
    if check.kind != "image":
        raise HTTPException(400, "This endpoint accepts png, jpg or jpeg images only.")
    return await _handle_image(check, content, user)


@app.post("/retrieve")
def retrieve_chunks(req: RetrieveRequest, user: dict = Depends(current_user)):
    return retrieve(req.query, top_k=req.top_k, source=req.source)


@app.post("/chat")
def chat(req: ChatRequest, user: dict = Depends(current_user)):
    # (a normal "def", not "async def": FastAPI runs it in a worker thread, which the MCP client needs)
    guard = check_input(req.message, has_upload=bool(req.extraction_id), history=req.history)  # 1. input guardrail
    logger.info("chat user=%s input=%s decision=%s", user["username"],
                mask_for_log(req.message)[:80], guard.reason)
    if not guard.allowed:
        return {"answer": guard.message, "found": False, "sources": [], "guard": guard.reason,
                "confidence": None, "evaluation": {}, "plan": {}, "trace": [], "seconds": 0}

    if req.extraction_id:
        found = get_extraction(req.extraction_id)
        if not found or not _visible(found, user):  # same message for "missing" and "not yours"
            raise HTTPException(404, "That uploaded image result was not found. Please upload the image again.")

    try:  # 2. planner -> retriever | mcp | vision -> reasoning -> evaluator
        state = run_graph(guard.masked_text, req.extraction_id or "")  # only masked text goes on
    except Exception:  # never show raw errors (they can contain keys or paths)
        logger.exception("Agent graph failed")
        raise HTTPException(500, "The assistant could not answer right now. Check the server log.")

    evaluation = state.get("evaluation", {})
    allowed = bool(state.get("allowed"))
    if allowed:
        reason = "ok"
    elif evaluation.get("decision") == "no_answer":
        reason = "no_evidence"
    else:
        reason = "output_blocked"
        logger.warning("Output blocked: %s", evaluation)
    return {
        "answer": state.get("final_answer", ""),
        "found": allowed,
        "sources": state.get("sources", []) if allowed else [],
        "guard": reason,
        "confidence": evaluation.get("confidence"),
        "evaluation": evaluation,  # for the dashboard
        "plan": state.get("plan", {}),
        "trace": state.get("trace", []),  # which agent ran and how long it took
        "seconds": state.get("total_seconds"),
    }


@app.get("/images")
def list_images(user: dict = Depends(current_user)):
    """The user's analysed images, newest first (the dropdown and the sidebar list use this)."""
    items = []
    files = sorted(config.EXTRACTIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        if not EXTRACTION_ID.match(path.stem):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _visible(data, user):
            items.append({"id": path.stem, "name": data.get("original_name") or data.get("file", "image"),
                          "image_type": data.get("image_type", "medical_image")})
        if len(items) >= 30:
            break
    return {"images": items}


@app.delete("/images/{extraction_id}")
def delete_image(extraction_id: str, user: dict = Depends(current_user)):
    data = get_extraction(extraction_id)  # also checks the id format (blocks path tricks)
    if not data or not _visible(data, user):
        raise HTTPException(404, "That image result was not found.")
    (config.EXTRACTIONS_DIR / f"{extraction_id}.json").unlink(missing_ok=True)
    (config.IMAGES_DIR / Path(data.get("file", "")).name).unlink(missing_ok=True)  # the saved picture too
    return {"deleted": extraction_id}


@app.get("/sources")
def get_sources(user: dict = Depends(current_user)):
    return {"sources": [{"name": n, "chunks": c} for n, c in list_sources().items()]}


@app.delete("/sources/{name}")
def delete_source(name: str, user: dict = Depends(current_user)):
    name = Path(name).name  # block path tricks like ../
    removed = remove_source(name)
    if removed == 0:
        raise HTTPException(404, f"'{name}' is not in the knowledge base.")
    (config.RAW_DIR / name).unlink(missing_ok=True)  # also delete the saved file
    return {"deleted": name, "chunks_removed": removed}
