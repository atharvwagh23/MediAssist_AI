"""
Creates 3 fake test images in data/test_images/ (no real patient data):
  prescription.png  - printed prescription
  lab_report.png    - printed lab report
  plain_photo.png   - a picture with no text (tests the "no readable text" path)

Run from the project root:
    python -m tests.make_test_images
"""

from PIL import Image, ImageDraw, ImageFont  # Pillow draws the pictures

from app import config

OUT = config.DATA_DIR / "test_images"  # output folder
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int):  # try common fonts, fall back to Pillow's built-in one
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_page(name: str, rows: list[tuple[int, int, str, bool]], size=(1400, 900)):
    """rows = (x, y, text, bold-ish). White page, black text."""
    page = Image.new("RGB", size, "white")
    pen = ImageDraw.Draw(page)
    for x, y, text, big in rows:
        pen.text((x, y), text, fill="black", font=font(40 if big else 30))
    page.save(OUT / name)
    print("created", OUT / name)


draw_page("prescription.png", [
    (60, 40, "CITY CARE HOSPITAL", True),
    (60, 100, "Dr. A. Sharma, MD (Medicine)     Ph: 9876543210", False),
    (60, 150, "Date: 2026-10-12      Patient: Ravi Kumar, 45 / M", False),
    (60, 220, "Diagnosis: Viral fever with throat infection", False),
    (60, 300, "Rx", True),
    (60, 370, "1. Tab Paracetamol 500 mg - 1 tablet TDS - 5 days", False),
    (60, 430, "2. Tab Azithromycin 250 mg - 1 tablet OD - 3 days", False),
    (60, 490, "3. Syp Cough Relief 10 ml - BD - 5 days", False),
    (60, 600, "Advice: Drink plenty of fluids. Take after food.", False),
    (60, 660, "Review after 5 days.", False),
])

draw_page("lab_report.png", [
    (60, 40, "CITY CARE DIAGNOSTIC LAB", True),
    (60, 100, "Patient ID: P00123          Date: 2026-10-14", False),
    (60, 170, "COMPLETE BLOOD COUNT", True),
    (60, 250, "Test", False), (420, 250, "Result", False), (600, 250, "Unit", False),
    (780, 250, "Reference Range", False), (1150, 250, "Flag", False),
    (60, 320, "Hemoglobin", False), (420, 320, "9.8", False), (600, 320, "g/dL", False),
    (780, 320, "12.0 - 16.0", False), (1150, 320, "Low", False),
    (60, 380, "WBC Count", False), (420, 380, "11500", False), (600, 380, "/uL", False),
    (780, 380, "4000 - 11000", False), (1150, 380, "High", False),
    (60, 440, "Platelets", False), (420, 440, "250000", False), (600, 440, "/uL", False),
    (780, 440, "150000 - 450000", False), (1150, 440, "Normal", False),
    (60, 500, "Fasting Glucose", False), (420, 500, "92", False), (600, 500, "mg/dL", False),
    (780, 500, "70 - 100", False), (1150, 500, "Normal", False),
    (60, 640, "Pathologist remarks: Mild anemia noted. Advised to consult physician.", False),
])

photo = Image.new("RGB", (800, 600), (30, 30, 30))  # dark picture with shapes, no text
pen = ImageDraw.Draw(photo)
pen.ellipse((150, 100, 650, 500), fill=(200, 200, 200))
pen.ellipse((300, 200, 500, 400), fill=(90, 90, 90))
photo.save(OUT / "plain_photo.png")
print("created", OUT / "plain_photo.png")