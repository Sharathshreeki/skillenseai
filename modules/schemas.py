
"""Shared data contracts for SkillLens AI.

These dataclasses define the common data structures used by
Member 2, Member 3, and Member 4.
Keep field names aligned with the team contract.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Mention:
    original_phrase: str
    evidence: str = ""
    span_start: int = 0
    span_end: int = 0
    source: str = "explicit"
    matched_via: str = "rule"

    def __post_init__(self):
        if self.span_start < 0 or self.span_end < self.span_start:
            raise ValueError("Mention span positions are invalid.")

        if self.source not in {"explicit", "inferred"}:
            raise ValueError(
                "Mention source must be 'explicit' or 'inferred'."
            )


@dataclass
class ExtractionResult:
    text: str
    method: str = "paste"
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        allowed_methods = {
            "txt",
            "docx",
            "pdf_text",
            "pdf_ocr",
            "image_ocr",
            "paste",
        }

        if self.method not in allowed_methods:
            raise ValueError(
                f"Invalid extraction method: {self.method}"
            )

        if not isinstance(self.text, str):
            raise TypeError("Extraction text must be a string.")

        if not isinstance(self.warnings, list):
            raise TypeError("Warnings must be a list.")


@dataclass
class Skill:
    skill_id: str
    skill: str
    category: str = "Technical"
    original_phrase: str = ""
    evidence: str = ""
    span_start: int = 0
    span_end: int = 0
    source: str = "explicit"
    requirement: str = "Uncertain"
    requirement_basis: str = "none"
    mapping_status: str = "unmapped"
    mapping_score: float = 0.0
    needs_review: bool = True
    review_reason: str = ""
    user_status: str = "pending"

    def __post_init__(self):
        allowed_categories = {"Technical", "Non-technical"}

        allowed_requirements = {
            "Mandatory",
            "Preferred",
            "Uncertain",
        }

        allowed_sources = {"explicit", "inferred"}

        allowed_mapping_statuses = {
            "exact",
            "alias",
            "review_alias",
            "fuzzy",
            "semantic",
            "unmapped",
        }

        allowed_user_statuses = {
            "pending",
            "approved",
            "rejected",
            "edited",
        }

        if self.category not in allowed_categories:
            raise ValueError(
                f"Invalid category: {self.category}"
            )

        if self.requirement not in allowed_requirements:
            raise ValueError(
                f"Invalid requirement: {self.requirement}"
            )

        if self.source not in allowed_sources:
            raise ValueError(
                f"Invalid source: {self.source}"
            )

        if self.mapping_status not in allowed_mapping_statuses:
            raise ValueError(
                f"Invalid mapping status: {self.mapping_status}"
            )

        if self.user_status not in allowed_user_statuses:
            raise ValueError(
                f"Invalid user status: {self.user_status}"
            )

        if self.span_start < 0 or self.span_end < self.span_start:
            raise ValueError("Skill span positions are invalid.")

        if not 0.0 <= self.mapping_score <= 1.0:
            raise ValueError(
                "mapping_score must be between 0 and 1."
            )

        if self.source == "inferred" and not self.needs_review:
            raise ValueError(
                "Inferred skills must be marked for review."
            )


@dataclass
class RequirementResult:
    skills: list[Skill] = field(default_factory=list)
    experience: dict[str, Any] = field(default_factory=dict)
    education: dict[str, Any] = field(default_factory=dict)
    certifications: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class JobProfile:
    job_id: str
    title: Optional[str] = None
    skills: list[Skill] = field(default_factory=list)
    experience: dict[str, Any] = field(default_factory=dict)
    education: dict[str, Any] = field(default_factory=dict)
    certifications: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    job_ids: list[str] = field(default_factory=list)
    matrix: list[dict[str, Any]] = field(default_factory=list)
    common_skills: list[Any] = field(default_factory=list)
    unique_skills: dict[str, list[Any]] = field(default_factory=dict)
    differences: list[dict[str, Any]] = field(default_factory=list)
    experience: dict[str, Any] = field(default_factory=dict)
    education: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)