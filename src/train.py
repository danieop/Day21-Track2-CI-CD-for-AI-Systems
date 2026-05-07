import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EVAL_THRESHOLD = 0.70
DEFAULT_MODEL_TYPE = "random_forest"


def _model_params(config: dict, model_type: str) -> dict:
    """
    Lay tham so cho tung loai model.

    Ho tro ca kieu cu (dict phang cho RandomForest) va kieu moi trong params.yaml:
    model_type + block random_forest/gradient_boosting/logistic_regression.
    """
    nested = config.get(model_type)
    if isinstance(nested, dict):
        return dict(nested)

    excluded = {
        "model_type",
        "experiments",
        "random_forest",
        "gradient_boosting",
        "logistic_regression",
    }
    return {key: value for key, value in config.items() if key not in excluded}


def build_model(model_type: str, params: dict):
    """Khoi tao model tu ten thuat toan va tham so tu params.yaml."""
    model_params = dict(params)
    random_state = model_params.pop("random_state", 42)

    if model_type == "random_forest":
        return RandomForestClassifier(**model_params, random_state=random_state)

    if model_type == "gradient_boosting":
        model_params.pop("n_jobs", None)
        model_params.pop("class_weight", None)
        return GradientBoostingClassifier(**model_params, random_state=random_state)

    if model_type == "logistic_regression":
        model_params.setdefault("max_iter", 1000)
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(**model_params, random_state=random_state),
        )

    raise ValueError(
        "Unsupported model_type. Use one of: random_forest, "
        "gradient_boosting, logistic_regression."
    )


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
    model_type: str | None = None,
    model_output_path: str = "models/model.pkl",
    metrics_output_path: str = "outputs/metrics.json",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    selected_model_type = model_type or params.get("model_type", DEFAULT_MODEL_TYPE)
    selected_params = _model_params(params, selected_model_type)

    with mlflow.start_run(run_name=selected_model_type):
        mlflow.log_param("model_type", selected_model_type)
        mlflow.log_params(selected_params)

        model = build_model(selected_model_type, selected_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(
            f"Model: {selected_model_type} | "
            f"Accuracy: {acc:.4f} | F1: {f1:.4f}"
        )

        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_path, "w") as f:
            json.dump(
                {
                    "model_type": selected_model_type,
                    "accuracy": acc,
                    "f1_score": f1,
                },
                f,
                indent=2,
            )

        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output_path)

    return float(acc)


def run_configured_experiments(
    config: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> dict:
    """
    Chay nhieu thuat toan va luu model tot nhat vao models/model.pkl.
    """
    experiments = config.get("experiments") or [
        config.get("model_type", DEFAULT_MODEL_TYPE)
    ]
    results = []

    for model_type in experiments:
        candidate_model = f"models/candidates/{model_type}.pkl"
        candidate_metrics = f"outputs/candidates/{model_type}.json"
        acc = train(
            config,
            data_path=data_path,
            eval_path=eval_path,
            model_type=model_type,
            model_output_path=candidate_model,
            metrics_output_path=candidate_metrics,
        )
        with open(candidate_metrics) as f:
            metrics = json.load(f)
        results.append(
            {
                "model_type": model_type,
                "accuracy": acc,
                "f1_score": metrics["f1_score"],
                "model_path": candidate_model,
            }
        )

    best = max(results, key=lambda item: (item["accuracy"], item["f1_score"]))
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    joblib.dump(joblib.load(best["model_path"]), "models/model.pkl")

    summary = {
        "best_model_type": best["model_type"],
        "accuracy": best["accuracy"],
        "f1_score": best["f1_score"],
        "results": results,
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Best model: {best['model_type']} | "
        f"Accuracy: {best['accuracy']:.4f} | F1: {best['f1_score']:.4f}"
    )
    return summary


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    run_configured_experiments(params)
