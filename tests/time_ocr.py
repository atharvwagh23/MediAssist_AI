"""
Measures OCR speed: model loading vs reading, at different image sizes.

Run from the project root:
    python -m tests.time_ocr data/test_images/lab_report.png
"""

import sys
import time

from app import config
from app.multimodal import image_processor, ocr

image = image_processor.load_image(sys.argv[1])  # the image to test

start = time.time()
ocr.get_reader()  # loads the EasyOCR model
print(f"Load EasyOCR model: {time.time() - start:.1f}s (happens once, when the server starts)")

ocr.extract_text(image_processor.prepare_for_ocr(image))  # warm-up run, not timed
for side in (1600, 1200, 1000, 800):  # try smaller and smaller images
    config.OCR_MAX_SIDE = side
    prepared = image_processor.prepare_for_ocr(image)
    start = time.time()
    result = ocr.extract_text(prepared)
    seconds = time.time() - start
    print(f"max side {side}: image {prepared.size}, {seconds:.1f}s, "
          f"lines={result['line_count']}, confidence={result['confidence']}")
    