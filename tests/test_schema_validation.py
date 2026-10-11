
import pytest

from modules.schemas import (
    ExtractionResult,
    Mention,
    Skill,
)

from dataclasses import asdict

from modules.schemas import (
    ComparisonResult,
    JobProfile,
    RequirementResult,
)


def test_valid_skill():
    skill = Skill(
        skill_id="python",
        skill="Python",
        requirement="Mandatory",
    )
    assert skill.requirement == "Mandatory"


def test_invalid_requirement():
    with pytest.raises(ValueError):
        Skill(
            skill_id="python",
            skill="Python",
            requirement="Required",
        )


def test_invalid_mapping_score():
    with pytest.raises(ValueError):
        Skill(
            skill_id="python",
            skill="Python",
            mapping_score=1.5,
        )


def test_inferred_skill_requires_review():
    with pytest.raises(ValueError):
        Skill(
            skill_id="python",
            skill="Python",
            source="inferred",
            needs_review=False,
        )


def test_invalid_mention_span():
    with pytest.raises(ValueError):
        Mention(
            original_phrase="Python",
            span_start=10,
            span_end=5,
        )


def test_invalid_extraction_method():
    with pytest.raises(ValueError):
        ExtractionResult(
            text="Example job description",
            method="invalid",
        )






def test_requirement_result_has_expected_contract():
    result = RequirementResult()

    assert set(asdict(result)) == {
        "skills",
        "experience",
        "education",
        "certifications",
        "warnings",
    }


def test_job_profile_has_expected_contract():
    profile = JobProfile(job_id="job-001", title="Data Analyst")

    assert set(asdict(profile)) == {
        "job_id",
        "title",
        "skills",
        "experience",
        "education",
        "certifications",
        "warnings",
    }


def test_comparison_result_has_expected_contract():
    result = ComparisonResult()

    assert set(asdict(result)) == {
        "job_ids",
        "matrix",
        "common_skills",
        "unique_skills",
        "differences",
        "experience",
        "education",
        "warnings",
    }


def test_skill_enum_values_follow_contract():
    valid_requirements = {"Mandatory", "Preferred", "Uncertain"}
    valid_categories = {"Technical", "Non-technical"}
    valid_sources = {"explicit", "inferred"}
    valid_mapping_statuses = {
        "exact",
        "alias",
        "review_alias",
        "fuzzy",
        "semantic",
        "unmapped",
    }
    valid_user_statuses = {"pending", "approved", "rejected", "edited"}

    skill = Skill(skill_id="python", skill="Python")

    assert skill.requirement in valid_requirements
    assert skill.category in valid_categories
    assert skill.source in valid_sources
    assert skill.mapping_status in valid_mapping_statuses
    assert skill.user_status in valid_user_statuses

