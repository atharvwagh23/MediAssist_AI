"""
MCP Agent: runs the database tools the Planner asked for, through the MCP client.

Safety checks before every call (in this order):
  1. role check: is this tool in ROLE_TOOLS for the logged-in user? (no role = nothing allowed)
  2. tool guard: allowlisted tool, plain short arguments, no SQL words
Calls that pass run in parallel (each MCP call starts its own server process).
Each entry in db_results: {"tool", "arguments", "status": ok|error|denied|blocked, "data"|"message"}.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from app import auth
from app.agents.state import AgentState, add_trace
from app.guardrails.input_guard import check_tool_call
from app.mcp import client

logger = logging.getLogger("mediassist.agents")


def _run(call: dict) -> dict:
    entry = {"tool": call["name"], "arguments": call["arguments"]}
    try:
        data = client.call_tool_sync(call["name"], call["arguments"])
    except Exception:  # never show raw errors (they can contain keys or paths)
        logger.exception("MCP call failed: %s", call["name"])
        return {**entry, "status": "error", "message": "The database could not be reached right now."}
    if isinstance(data, dict) and data.get("error"):  # tool-level message, already safe to show
        return {**entry, "status": "error", "message": str(data["error"])}
    return {**entry, "status": "ok", "data": data}


def mcp_agent(state: AgentState) -> dict:
    started = time.time()
    plan = state["plan"]
    allowed_tools = auth.ROLE_TOOLS.get(state.get("role") or "", [])

    results = [None] * len(plan["calls"])  # one slot per planned call, so the plan order is kept
    runnable = []  # (slot, call)
    for slot, call in enumerate(plan["calls"]):
        if call["name"] not in allowed_tools:
            results[slot] = {"tool": call["name"], "arguments": call["arguments"], "status": "denied",
                             "message": f"The role '{state.get('role') or 'unknown'}' cannot use {call['name']}."}
            continue
        ok, reason = check_tool_call(call["name"], call["arguments"])
        if not ok:
            results[slot] = {"tool": call["name"], "arguments": call["arguments"],
                             "status": "blocked", "message": reason}
            continue
        runnable.append((slot, call))

    if runnable:
        with ThreadPoolExecutor(max_workers=min(len(runnable), 4)) as pool:
            for (slot, _), entry in zip(runnable, pool.map(_run, [c for _, c in runnable])):
                results[slot] = entry

    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("ok", "error", "denied", "blocked")}
    note = "no tool calls" if not plan["calls"] else ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    return {"db_results": results, "trace": add_trace(state, "mcp", started, note)}
