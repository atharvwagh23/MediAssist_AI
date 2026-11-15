"""
FastAPI backend for MediAssist AI.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000/docs to try every endpoint.

Endpoints:
    GET    /health         is the API alive, what is indexed
    POST   /upload         upload a document (or image) and index it
    POST   /retrieve       semantic search, returns chunks with scores
    POST   /chat           guardrails + retrieve + LLM answer with sources
    GET    /sources        list indexed files
    DELETE /sources/{name} remove a file from the knowledge base
"""

import logging  # server-side error logs
from pathlib import Path
from uuid import uuid4  # random file names for images

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.guardrails.file_guard import check_document_text, check_file
from app.guardrails.input_guard import check_input
from app.guardrails.output_guard import check_output
from app.guardrails.pii import mask_for_log
from app.llm import get_llm
from app.prompts import RAG_ANSWER_PROMPT
from app.rag.loader import load_document
from app.rag.retriever import retrieve
from app.rag.vector_store import add_file, list_sources, remove_source

logger = logging.getLogger("mediassist")

NO_RESULT_MESSAGE = (
    "I couldn't find information about this in the hospital knowledge base. "
    "Please rephrase your question or contact the hospital help desk."
)

app = FastAPI(title="MediAssist AI API", version="0.1.0")


# ---------- request shapes (FastAPI validates these automatically) ----------

class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown fields
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    source: str | None = None  # optional: search inside one file only


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


# ---------- endpoints ----------

@app.get("/health")
def health():
    sources = list_sources()
    return {"status": "ok", "indexed_chunks": sum(sources.values()), "sources": list(sources)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    check = check_file(file.filename, content)  # type, size, real file type, macros, scripts...
    if not check.ok:
        raise HTTPException(check.status, check.message)

    if check.kind == "image":  # images are only stored for now
        saved = config.IMAGES_DIR / f"{uuid4().hex}.{check.ext}"
        saved.write_bytes(content)
        return {"kind": "image", "saved_as": saved.name,
                "message": "Image saved. Image analysis is added in the multimodal step."}

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


@app.post("/retrieve")
def retrieve_chunks(req: RetrieveRequest):
    return retrieve(req.query, top_k=req.top_k, source=req.source)


@app.post("/chat")
def chat(req: ChatRequest):
    guard = check_input(req.message)  # 1. input guardrail (injection, PII, vague, medical scope)
    logger.info("chat input=%s decision=%s", mask_for_log(req.message)[:80], guard.reason)
    if not guard.allowed:
        return {"answer": guard.message, "found": False, "sources": [], "guard": guard.reason}
    question = guard.masked_text  # only the masked text goes to the LLM

    result = retrieve(question, top_k=req.top_k)
    if not result["found"]:  # nothing relevant: do not guess
        return {"answer": NO_RESULT_MESSAGE, "found": False, "sources": [], "guard": "no_evidence"}

    context = "\n\n".join(
        f"[{r['source']}, page {r['page']}]\n{r['text']}" for r in result["results"]
    )
    try:
        chain = ChatPromptTemplate.from_template(RAG_ANSWER_PROMPT) | get_llm()
        answer = chain.invoke({"question": question, "context": context}).content.strip()
    except Exception:  # never show raw errors (they can contain keys or paths)
        logger.exception("LLM call failed")
        top = result["results"][0]
        answer = ("The language model is not available right now. "
                  f"Most relevant passage ({top['source']}, page {top['page']}):\n\n{top['text']}")

    sources, seen = [], set()  # one entry per file+page
    for r in result["results"]:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": r["source"], "page": r["page"], "score": r["final_score"]})

    out = check_output(answer, sources, evidence=context)  # 2. output guardrail
    if not out.allowed:
        logger.warning("Output blocked: %s", out.issues)
        return {"answer": out.answer, "found": False, "sources": [], "guard": "output_blocked"}
    return {"answer": out.answer, "found": True, "sources": sources, "guard": "ok"}


@app.get("/sources")
def get_sources():
    return {"sources": [{"name": n, "chunks": c} for n, c in list_sources().items()]}


@app.delete("/sources/{name}")
def delete_source(name: str):
    name = Path(name).name  # block path tricks like ../
    removed = remove_source(name)
    if removed == 0:
        raise HTTPException(404, f"'{name}' is not in the knowledge base.")
    (config.RAW_DIR / name).unlink(missing_ok=True)  # also delete the saved file
    return {"deleted": name, "chunks_removed": removed}