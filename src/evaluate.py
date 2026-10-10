"""Evaluate a trained churn model: metrics, ROC/PR curves, feature importance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay,
                             classification_report, confusion_matrix)  # noqa: E402

from data import TARGET  # noqa: E402


def evaluate(model_path: Path = Path("artifacts/model.joblib"),
             test_csv: Path = Path("data/test.csv"),
             out_dir: Path = Path("artifacts")) -> None:
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_csv)
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(classification_report(y_test, pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=120)
    print(f"Saved ROC/PR curves to {out_dir / 'curves.png'}")

    # Model-agnostic importance: permutation_importance shuffles the RAW
    # input columns (pre-preprocessing), so index by those, not the
    # one-hot expanded names.
    from sklearn.inspection import permutation_importance
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    result = permutation_importance(
        model, sample, y_test.loc[sample.index],
        n_repeats=5, random_state=42, n_jobs=-1, scoring="roc_auc",
    )
    names = list(X_test.columns)
    top = (pd.Series(result.importances_mean, index=names)
           .sort_values(ascending=False).head(15))
    print("\nTop 15 features (permutation importance, ROC AUC):\n",
          top.round(4).to_string())
    (out_dir / "feature_importance.json").write_text(top.round(4).to_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the churn model.")
    parser.add_argument("--model", type=str, default="artifacts/model.joblib")
    parser.add_argument("--test-csv", type=str, default="data/test.csv")
    args = parser.parse_args()
    evaluate(Path(args.model), Path(args.test_csv))


if __name__ == "__main__":
    main()
