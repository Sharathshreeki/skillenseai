#!/usr/bin/env python3
"""Check Python OCR package and native Tesseract availability."""
import shutil
import sys

print("Python interpreter:", sys.executable)
print("Tesseract executable:", shutil.which("tesseract") or "NOT FOUND")
try:
    import pytesseract
    print("pytesseract package: OK")
    try:
        print("Tesseract version:", pytesseract.get_tesseract_version())
    except Exception as exc:
        print("Tesseract binary check FAILED:", exc)
        raise SystemExit(2)
except ImportError as exc:
    print("pytesseract package: MISSING", exc)
    raise SystemExit(1)
