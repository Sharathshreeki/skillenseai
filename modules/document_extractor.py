"""Document and OCR extraction for SkillLens AI.

Supports TXT, DOCX, searchable/scanned PDF, and common image formats.
OCR engine preference: PaddleOCR (if installed), then Tesseract via pytesseract.
"""
from __future__ import annotations

import io
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from modules.schemas import ExtractionResult

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _preprocess(image: Image.Image) -> Image.Image:
    """Normalize image orientation and improve contrast for OCR."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Upscale small screenshots/scans for better character recognition.
    if image.width < 1400:
        scale = min(2.0, 1400 / max(image.width, 1))
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    image = ImageOps.grayscale(image)
    image = ImageEnhance.Contrast(image).enhance(1.6)
    image = image.filter(ImageFilter.SHARPEN)
    return image


@lru_cache(maxsize=1)
def _paddle_engine():
    from paddleocr import PaddleOCR
    # PaddleOCR 3.x API. If the installed version only supports the older API,
    # the caller will fall back to Tesseract.
    return PaddleOCR(lang="en", use_textline_orientation=True, enable_mkldnn=False, show_log=False)


def _texts_from_paddle_result(result: Any) -> list[str]:
    texts: list[str] = []
    if not result:
        return texts
    for page in result:
        # PaddleOCR 3.x returns a result object with a JSON payload.
        payload = None
        try:
            payload = page.json
            if callable(payload):
                payload = payload()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            payload = payload.get("res", payload)
            rec_texts = payload.get("rec_texts") or payload.get("texts") or []
            texts.extend(str(t).strip() for t in rec_texts if str(t).strip())
            if rec_texts:
                continue
        # PaddleOCR 2.x legacy shape: [[box, (text, confidence)], ...]
        try:
            for line in page:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    recognition = line[1]
                    if isinstance(recognition, (list, tuple)) and recognition:
                        value = str(recognition[0]).strip()
                        if value:
                            texts.append(value)
        except (TypeError, ValueError):
            pass
    return texts


def _ocr_image(image: Image.Image) -> str:
    prepared = _preprocess(image)
    errors: list[str] = []

    # PaddleOCR first; support the current predict API and the legacy ocr API.
    try:
        import numpy as np
        engine = _paddle_engine()
        if hasattr(engine, "predict"):
            result = engine.predict(input=np.asarray(prepared))
        else:
            result = engine.ocr(np.asarray(prepared), cls=True)
        lines = _texts_from_paddle_result(result)
        if lines:
            return "\n".join(lines)
    except Exception as exc:
        errors.append(f"PaddleOCR: {exc}")

    # Tesseract is a useful lightweight fallback, but its native binary must be installed.
    try:
        import pytesseract
        # Streamlit launched from an IDE may not inherit Homebrew's PATH. Probe
        # common macOS and Linux install locations before invoking Tesseract.
        configured = os.environ.get("TESSERACT_CMD")
        candidates = [
            configured,
            shutil.which("tesseract"),
            "/opt/homebrew/bin/tesseract",  # Apple Silicon Homebrew
            "/usr/local/bin/tesseract",     # Intel Homebrew
            "/usr/bin/tesseract",           # Debian/Ubuntu
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
        text = pytesseract.image_to_string(prepared, lang="eng", config="--psm 6").strip()
        if text:
            return text
        errors.append("Tesseract returned no text")
    except Exception as exc:
        errors.append(f"Tesseract: {exc}")

    detail = "; ".join(errors[-2:])
    raise RuntimeError(
        "OCR could not extract text. Check that an OCR engine is installed and that the image contains readable text.\n\n"
        "macOS fix (recommended):\n"
        "1. In the same project environment used to run Streamlit, run: "
        "python -m pip install pytesseract\n"
        "2. Install the native OCR program once with: brew install tesseract\n"
        "3. Verify it with: python -c \"import pytesseract; "
        "print(pytesseract.get_tesseract_version())\"\n"
        "4. Restart the app using: python -m streamlit run app.py\n\n"
        "Note: installing the Python package alone is not enough; the native "
        "'tesseract' executable must also be installed.\n"
        f"Diagnostic details: {detail}"
    )


def extract_text(file: Any) -> ExtractionResult:
    """Extract text from a Streamlit UploadedFile or a file-like object."""
    name = getattr(file, "name", "upload.txt")
    suffix = Path(name).suffix.lower()
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()

    if suffix == ".txt":
        text = raw.decode("utf-8-sig", errors="replace").strip()
        return ExtractionResult(text=text, method="txt")

    if suffix == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(raw))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        # Include table content often used for job requirements.
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))
        text = "\n".join(paragraphs).strip()
        return ExtractionResult(text=text, method="docx")

    if suffix == ".pdf":
        import fitz
        with fitz.open(stream=raw, filetype="pdf") as pdf:
            page_texts = [page.get_text("text").strip() for page in pdf]
            native_text = "\n\n".join(t for t in page_texts if t)
            # A low character count usually indicates an image-only/scanned PDF.
            if len(native_text.replace("\n", "").strip()) >= 30:
                return ExtractionResult(text=native_text, method="pdf_text")
            ocr_pages: list[str] = []
            for index, page in enumerate(pdf):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                page_text = _ocr_image(image)
                if page_text.strip():
                    ocr_pages.append(f"[Page {index + 1}]\n{page_text.strip()}")
            text = "\n\n".join(ocr_pages).strip()
            if not text:
                raise ValueError("No readable text found in PDF, including OCR output.")
            return ExtractionResult(text=text, method="pdf_ocr")

    if suffix in IMAGE_SUFFIXES:
        image = Image.open(io.BytesIO(raw))
        text = _ocr_image(image).strip()
        if not text:
            raise ValueError("OCR finished but no readable text was detected in this image.")
        return ExtractionResult(text=text, method="image_ocr")

    raise ValueError(f"Unsupported file type: {name}")
