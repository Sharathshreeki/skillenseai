
"""Evaluate extracted skills against reference skills in the labelled dataset."""

import ast
import json
import re
from pathlib import Path

import pandas as pd

from modules.pipeline import analyze_job


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "skilllens_dataset.csv"
SAMPLE_SIZE = 100
RANDOM_STATE = 42


def normalize_skill(skill: str) -> str:
    """Normalize case and whitespace for fair string comparison."""
    return re.sub(r"\s+", " ", skill.strip().casefold())


def parse_reference_skills(value) -> list[str]:
    """Parse a dataset cell containing a Python-list-style skill string."""
    if isinstance(value, list):
        skills = value
    elif isinstance(value, str):
        try:
            skills = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"Could not parse reference skill list: {value!r}")
    else:
        raise ValueError(f"Unexpected reference skill format: {type(value).__name__}")

    if not isinstance(skills, list):
        raise ValueError("Reference skills must be stored as a list.")

    return [
        skill.strip()
        for skill in skills
        if isinstance(skill, str) and skill.strip()
    ]


def calculate_metrics(reference: set[str], predicted: set[str]) -> dict:
    """Calculate micro-style metrics for one job description."""
    true_positives = len(reference & predicted)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"job_id", "job_title", "job_description", "job_skill_set"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    df = df.dropna(subset=["job_description", "job_skill_set"])

    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )

    totals = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    per_job_results = []

    for _, row in sample.iterrows():
        reference_list = parse_reference_skills(row["job_skill_set"])
        reference = {normalize_skill(skill) for skill in reference_list}

        profile = analyze_job(
            text=str(row["job_description"]),
            job_id=str(row["job_id"]),
            title=str(row["job_title"]),
        )

        predicted = {
            normalize_skill(skill.skill)
            for skill in profile.skills
            if skill.skill.strip()
        }

        metrics = calculate_metrics(reference, predicted)

        for key in totals:
            totals[key] += metrics[key]

        per_job_results.append({
            "job_id": row["job_id"],
            "job_title": row["job_title"],
            "reference_count": len(reference),
            "predicted_count": len(predicted),
            **metrics,
            "warnings": json.dumps(profile.warnings),
        })

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    output_dir = ROOT / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / "skill_extraction_results.csv"
    pd.DataFrame(per_job_results).to_csv(results_path, index=False)

    print(f"Dataset rows: {len(df)}")
    print(f"Sample size: {len(sample)}")
    print(f"Random seed: {RANDOM_STATE}")
    print("\nAggregate skill-extraction metrics")
    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-score:        {f1:.4f}")
    print(f"\nPer-job results saved to: {results_path}")


if __name__ == "__main__":
    main()
