"""Central configuration for SkillLens AI."""

import os

OCR_ENABLED = os.getenv("OCR_ENABLED", "false").lower() == "true"
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
