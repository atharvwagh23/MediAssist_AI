"""Retriever Agent: searches the hospital documents (FAISS + BM25) when the plan asks for it."""

import time

from app.agents.state import AgentState, add_trace
from app.rag.retriever import retrieve


def retriever_agent(state: AgentState) -> dict:
    started = time.time()
    if not state["plan"]["use_documents"]:
        return {"doc_results": [], "trace": add_trace(state, "retriever", started, "skipped (not needed)")}
    result = retrieve(state["question"])
    chunks = result["results"] if result["found"] else []
    note = f"{len(chunks)} chunks" + (f", best score {chunks[0]['final_score']}" if chunks else ", nothing relevant")
    return {"doc_results": chunks, "trace": add_trace(state, "retriever", started, note)}
