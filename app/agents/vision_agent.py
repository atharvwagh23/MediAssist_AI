"""
Vision Agent (the multimodal agent): loads the saved result of an uploaded image.

The slow work (OCR, BLIP, extraction) already happened at upload time in step 8,
so chat never runs OCR again. This agent only reads the saved JSON.
"""

import time

from app.agents.state import AgentState, add_trace
from app.multimodal.report_analyzer import get_extraction


def vision_agent(state: AgentState) -> dict:
    started = time.time()
    if not state["plan"]["use_image"]:
        return {"trace": add_trace(state, "vision", started, "skipped (not needed)")}
    image = get_extraction(state.get("extraction_id", "")) or state.get("image")
    if not image:
        return {"image": {}, "trace": add_trace(state, "vision", started, "image result not found")}
    note = f"{image['image_type']}, {len(image['medicines'])} medicines, {len(image['observations'])} observations"
    return {"image": image, "trace": add_trace(state, "vision", started, note)}
