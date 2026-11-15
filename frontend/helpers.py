"""Small pure functions used by the Streamlit page (no Streamlit code, so they are easy to test)."""

import time


def push_turn(turns: list[dict], turn: dict, limit: int) -> list[dict]:
    """Add a question/answer pair. Only the last `limit` pairs are kept (oldest dropped first: FIFO)."""
    return (list(turns) + [turn])[-limit:]


def history_for_backend(turns: list[dict], count: int = 3) -> list[str]:
    """The last few user questions, sent so follow-up questions make sense."""
    return [t["question"][:300] for t in turns[-count:]]


def trace_rows(trace: list[dict]) -> list[dict]:
    """Agent execution flow as table rows."""
    return [{"Agent": t["agent"], "Seconds": t["seconds"], "What it did": t["note"]} for t in trace]


def source_lines(sources: list[dict]) -> list[str]:
    """One readable line per source."""
    lines = []
    for s in sources:
        kind = s.get("type", "document")
        if kind == "database":
            lines.append(f"Database: {s['source']}" + (f" (patient {s['detail']})" if s.get("detail") else ""))
        elif kind == "image":
            lines.append(f"Uploaded image: {s['source']}")
        else:
            score = f", score {s['score']}" if s.get("score") is not None else ""
            lines.append(f"{s['source']} (page {s.get('page')}{score})")
    return lines


def show(value, digits: int = 2) -> str:
    """Metric text: a number, or 'n/a' when the value is missing."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def typewriter(text: str):
    """Yield the answer a few words at a time so it appears gradually (the answer is already complete)."""
    words = text.split(" ")
    delay = min(0.02, 2.0 / max(len(words), 1))  # never more than about 2 seconds in total
    for i in range(0, len(words), 2):
        yield " ".join(words[i:i + 2]) + " "
        time.sleep(delay)


def decision_text(evaluation: dict) -> str:
    return {
        "allowed": "Allowed (rules and judge passed)",
        "allowed_rules_only": "Allowed (rules passed, judge unavailable)",
        "blocked_by_rules": "Blocked by the safety rules",
        "blocked_by_judge": "Blocked by the judge",
        "blocked_off_scope": "Blocked: not a medical answer",
        "no_answer": "No answer (nothing to answer from, or not allowed for this role)",
        "error": "Error while checking",
    }.get(evaluation.get("decision", ""), "n/a")
