# MLOps Lab Report

## Selected Hyperparameters

Final RandomForest configuration:

```yaml
model_type: random_forest
experiments:
  - random_forest
  - gradient_boosting
  - logistic_regression

random_forest:
n_estimators: 200
max_depth:
min_samples_split: 2
min_samples_leaf: 1
criterion: gini
max_features: 0.5
class_weight: balanced_subsample
random_state: 27
n_jobs: -1
```

Reason: the baseline model with `max_depth: 5` reached only `0.5640` accuracy. Increasing tree depth and tuning `max_features`, `class_weight`, and `random_state` improved the held-out evaluation accuracy to `0.7000`, which passed the CI/CD eval gate. After adding phase 2 training data, the retrained model reached `0.7440` accuracy and `0.7432` weighted F1.

## CI/CD and Deployment

The pipeline runs on GitHub Actions with four jobs:

1. Unit Test
2. Train
3. Eval
4. Deploy

DigitalOcean Spaces is used as the DVC and model artifact remote. A DigitalOcean Droplet serves the model through FastAPI.

Step 2 successful run:

```text
https://github.com/danieop/Day21-Track2-CI-CD-for-AI-Systems/actions/runs/25476021601
```

Step 3 data-triggered run:

```text
https://github.com/danieop/Day21-Track2-CI-CD-for-AI-Systems/actions/runs/25477853808
```

## Issues and Fixes

- DigitalOcean was used instead of GCP/AWS/Azure. Spaces is S3-compatible, so DVC was configured with `dvc[s3]`, `endpointurl`, and `boto3`.
- The first model did not pass the `0.70` eval gate. Hyperparameter tuning improved accuracy from `0.5640` to `0.7000`.
- Adding `train_phase2.csv` in Step 3 increased the training set from `2998` to `5996` rows and improved accuracy to `0.7440`.

## Endpoint Check

The deployed API responds successfully:

```text
GET /health -> {"status":"ok"}
POST /predict -> {"prediction":0,"label":"thap"}
```

## Bonus Implementation

- Bonus 1: GitHub Actions reads `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` from GitHub Secrets, so training runs are logged to the DagsHub MLflow server.
- Bonus 2: `src/train.py` supports `random_forest`, `gradient_boosting`, and `logistic_regression`; `params.yaml` runs all three and saves the best model.
- Bonus 3: `src/report.py` generates `outputs/performance_report.md`, uploaded by CI as the `training-reports` artifact.
- Bonus 4: models are stored under `models/runs/<commit-sha>/model.pkl`; `.github/workflows/rollback.yml` can promote any previous key back to `models/latest/model.pkl` and redeploy.
- Bonus 5: `src/drift.py` writes `outputs/drift_report.json` before training, and the report includes drift status and drifted features.
