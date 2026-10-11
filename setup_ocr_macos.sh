#!/usr/bin/env bash
# Set up SkillLens AI OCR on macOS and verify it uses one consistent Python environment.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v tesseract >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Tesseract was not found and Homebrew is unavailable. Install Homebrew from https://brew.sh, then rerun this script."
    exit 1
  fi
  echo "Installing native Tesseract OCR..."
  brew install tesseract
else
  echo "Tesseract executable found: $(command -v tesseract)"
fi

if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  echo "Creating a project virtual environment with Python 3.11..."
  python3.11 -m venv "$PROJECT_DIR/.venv"
  PYTHON="$PROJECT_DIR/.venv/bin/python"
else
  echo "Python 3.11 was not found; creating the environment with python3."
  python3 -m venv "$PROJECT_DIR/.venv"
  PYTHON="$PROJECT_DIR/.venv/bin/python"
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$PYTHON" - <<'PY'
import pytesseract
print("Python package: pytesseract is installed")
print("Native Tesseract version:", pytesseract.get_tesseract_version())
PY

echo
echo "OCR setup verified. Start SkillLens AI with:"
echo "  cd \"$PROJECT_DIR\""
echo "  .venv/bin/python -m streamlit run app.py"
