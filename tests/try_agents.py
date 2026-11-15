"""
Runs ONE real question through all six agents (real LLM, real database, real documents)
and prints the answer, the agent trace, the evaluation and the timings.

Run from the project root:
    python -m tests.try_agents "What is the admission process?"
    python -m tests.try_agents "Show lab results for P00001" --role doctor
    python -m tests.try_agents "Show billing for P00001" --role doctor        (should be denied)
    python -m tests.try_agents "Summarise this report" --image-id <id printed by try_image>
"""

import argparse
import json

from app.agents.graph import run_graph
from app.guardrails.input_guard import check_input

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--role", default="admin")
parser.add_argument("--image-id", default="")
args = parser.parse_args()

guard = check_input(args.question, has_upload=bool(args.image_id))
if not guard.allowed:
    raise SystemExit(f"Input guardrail stopped it ({guard.reason}): {guard.message}")

state = run_graph(guard.masked_text, args.role, args.image_id)
print("\n==== ANSWER ====")
print(state["final_answer"])
print("\n==== PLAN ====")
print(json.dumps({k: v for k, v in state["plan"].items() if k != "subtasks"}, indent=2))
print("\n==== TRACE (agent, seconds, note) ====")
for t in state["trace"]:
    print(f"  {t['agent']:<10} {t['seconds']:>5}s  {t['note']}")
print("\n==== EVALUATION ====")
print(json.dumps(state["evaluation"], indent=2))
print(f"\nallowed={state['allowed']}  total={state['total_seconds']}s  (the 15 s limit applies to chat)")
