import os
import json
import numpy as np
import pandas as pd
from src.drift import check_data_drift
from src.report import generate_report
from src.train import build_model, run_configured_experiments, train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi cloud storage hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_model_file_created(tmp_path):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")


def test_build_model_supports_multiple_algorithms():
    """Kiem tra Bonus 2: train.py ho tro nhieu thuat toan."""
    assert build_model("random_forest", {"n_estimators": 5}) is not None
    assert build_model("gradient_boosting", {"n_estimators": 5}) is not None
    assert build_model("logistic_regression", {"max_iter": 100}) is not None


def test_run_configured_experiments_selects_best_model(tmp_path):
    """Kiem tra viec chay nhieu model va luu metrics tong hop."""
    train_path, eval_path = _make_temp_data(tmp_path)
    config = {
        "experiments": ["random_forest", "logistic_regression"],
        "random_forest": {"n_estimators": 5, "max_depth": 3},
        "logistic_regression": {"max_iter": 100},
    }

    summary = run_configured_experiments(
        config,
        data_path=train_path,
        eval_path=eval_path,
    )

    assert "best_model_type" in summary
    assert len(summary["results"]) == 2
    assert os.path.exists("models/model.pkl")


def test_drift_report_and_performance_report_created(tmp_path):
    """Kiem tra Bonus 3 va Bonus 5 tao report tu dong."""
    train_path, eval_path = _make_temp_data(tmp_path)
    drift_path = str(tmp_path / "drift_report.json")
    metrics_path = str(tmp_path / "metrics.json")
    report_path = str(tmp_path / "performance_report.md")

    drift = check_data_drift(
        reference_path=eval_path,
        current_path=train_path,
        output_path=drift_path,
    )
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_model_type": "random_forest",
                "accuracy": 0.8,
                "f1_score": 0.79,
                "results": [
                    {
                        "model_type": "random_forest",
                        "accuracy": 0.8,
                        "f1_score": 0.79,
                    }
                ],
            },
            f,
        )

    generate_report(
        metrics_path=metrics_path,
        drift_path=drift_path,
        output_path=report_path,
    )

    assert "feature_scores" in drift
    assert os.path.exists(report_path)
