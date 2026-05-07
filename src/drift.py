import json
from pathlib import Path

import pandas as pd


def _numeric_features(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column != "target"
    ]


def check_data_drift(
    reference_path: str = "data/eval.csv",
    current_path: str = "data/train_phase1.csv",
    output_path: str = "outputs/drift_report.json",
    threshold: float = 0.25,
    fail_on_drift: bool = False,
) -> dict:
    """
    Kiem tra drift don gian bang chenhlech mean da chuan hoa theo std reference.
    """
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    features = _numeric_features(reference)

    scores = {}
    for feature in features:
        ref_std = reference[feature].std()
        if ref_std == 0 or pd.isna(ref_std):
            score = 0.0
        else:
            score = abs(current[feature].mean() - reference[feature].mean()) / ref_std
        scores[feature] = float(score)

    drifted_features = [
        feature for feature, score in scores.items() if score >= threshold
    ]
    report = {
        "reference_path": reference_path,
        "current_path": current_path,
        "threshold": threshold,
        "drift_detected": bool(drifted_features),
        "max_drift_score": max(scores.values()) if scores else 0.0,
        "drifted_features": drifted_features,
        "feature_scores": scores,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if fail_on_drift and report["drift_detected"]:
        raise SystemExit("Data drift detected. Check outputs/drift_report.json.")

    return report


if __name__ == "__main__":
    check_data_drift()
