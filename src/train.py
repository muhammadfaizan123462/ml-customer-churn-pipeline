"""Train the churn model: preprocessing + gradient boosting with tuning.

Pipeline:
    custom feature engineering -> ColumnTransformer (scale numerics,
    one-hot categoricals) -> HistGradientBoostingClassifier tuned with
    RandomizedSearchCV under stratified cross-validation.

Artifacts written to artifacts/:
    model.joblib      fitted pipeline (preprocessing + estimator)
    metrics.json      test-set metrics
    params.json       best hyperparameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from features import ChargeRatioAdder, RareCategoryGrouper, TenureBinner

ARTIFACTS = Path("artifacts")

PARAM_DIST = {
    "clf__learning_rate": loguniform(0.03, 0.3),
    "clf__max_iter": randint(100, 400),
    "clf__max_leaf_nodes": randint(15, 63),
    "clf__min_samples_leaf": randint(10, 60),
    "clf__l2_regularization": loguniform(1e-3, 10.0),
}


def build_pipeline() -> Pipeline:
    numeric = NUMERIC_FEATURES + ["avg_charge_per_month", "high_spend_short_tenure"]
    categorical = CATEGORICAL_FEATURES + ["tenure_segment"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(steps=[
        ("tenure_bin", TenureBinner()),
        ("charge_ratio", ChargeRatioAdder()),
        ("rare_group", RareCategoryGrouper(columns=CATEGORICAL_FEATURES)),
        ("prep", preprocessor),
        ("clf", HistGradientBoostingClassifier(random_state=42)),
    ])


def train(data_dir: Path = Path("data"), n_iter: int = 20) -> dict:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    search = RandomizedSearchCV(
        build_pipeline(),
        PARAM_DIST,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "best_cv_roc_auc": round(float(search.best_score_), 4),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "params.json").write_text(
        json.dumps(search.best_params_, indent=2, default=str)
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="RandomizedSearchCV iterations")
    args = parser.parse_args()
    train(n_iter=args.n_iter)


if __name__ == "__main__":
    main()
