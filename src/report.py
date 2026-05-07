import json
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def generate_report(
    metrics_path: str = "outputs/metrics.json",
    drift_path: str = "outputs/drift_report.json",
    output_path: str = "outputs/performance_report.md",
) -> str:
    metrics = _load_json(metrics_path)
    drift = _load_json(drift_path) if Path(drift_path).exists() else None

    best_model = metrics.get("best_model_type") or metrics.get("model_type", "unknown")
    accuracy = metrics.get("accuracy", 0.0)
    f1_score = metrics.get("f1_score", 0.0)

    lines = [
        "# Automatic Model Performance Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Best Model",
        "",
        f"- Model type: `{best_model}`",
        f"- Accuracy: `{accuracy:.4f}`",
        f"- Weighted F1: `{f1_score:.4f}`",
    ]

    if metrics.get("results"):
        lines.extend(["", "## Candidate Comparison", ""])
        lines.append("| Model | Accuracy | Weighted F1 |")
        lines.append("| --- | ---: | ---: |")
        for item in metrics["results"]:
            lines.append(
                "| {model_type} | {accuracy:.4f} | {f1_score:.4f} |".format(
                    **item
                )
            )

    if drift:
        lines.extend(
            [
                "",
                "## Data Drift Check",
                "",
                f"- Status: `{'DRIFT' if drift['drift_detected'] else 'OK'}`",
                f"- Max feature drift score: `{drift['max_drift_score']:.4f}`",
                f"- Drift threshold: `{drift['threshold']:.4f}`",
                f"- Drifted features: `{', '.join(drift['drifted_features']) or 'none'}`",
            ]
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_report()
