"""
Try the multimodal pipeline on ONE real image and print everything.

Run from the project root:
    python -m tests.try_image data/test_images/prescription.png
    python -m tests.try_image data/test_images/prescription.png --vision always
    python -m tests.try_image data/test_images/prescription.png --ocr-only
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles can choke on odd characters

from app import config
from app.multimodal import image_processor
from app.multimodal.report_analyzer import analyze_image

parser = argparse.ArgumentParser()
parser.add_argument("image")  # path to the image
parser.add_argument("--vision", choices=["auto", "always", "off"], help="override VISION_MODE")
parser.add_argument("--ocr-only", action="store_true", help="skip the LLM, show OCR lines only")
args = parser.parse_args()

if args.vision:
    config.VISION_MODE = args.vision  # change the setting just for this run

if args.ocr_only:
    info = image_processor.process_image(args.image)
    print(f"type={info['image_type']}  lines={info['line_count']}  confidence={info['ocr_confidence']}")
    print(f"ocr={info['ocr_seconds']}s  vision={info['vision_seconds']}s  caption={info['vision_caption']}")
    print("---- OCR lines (confidence | text) ----")
    for line in info["lines"]:
        print(f"{line['confidence']:.2f} | {line['text']}")
else:
    result = analyze_image(args.image)
    print(result["summary"])
    print("\n---- warnings ----")
    for warning in result["warnings"] or ["(none)"]:
        print("-", warning)
    print("\n---- timings (seconds) ----")
    print(result["timings"])
    print("saved id:", result["id"])