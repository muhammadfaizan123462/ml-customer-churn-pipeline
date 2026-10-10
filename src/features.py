"""Feature engineering: custom transformers used inside the sklearn Pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TenureBinner(BaseEstimator, TransformerMixin):
    """Bin raw tenure months into interpretable customer segments."""

    def __init__(self, bins: tuple = (0, 12, 24, 48, 100),
                 labels: tuple = ("new", "establishing", "loyal", "veteran")):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):  # noqa: ANN001, ANN202
        return self

    def transform(self, X):  # noqa: ANN001, ANN202
        X = X.copy()
        X["tenure_segment"] = pd.cut(
            X["tenure_months"], bins=self.bins, labels=self.labels
        ).astype(str)
        return X


class ChargeRatioAdder(BaseEstimator, TransformerMixin):
    """Add average-charge-per-tenure-month ratio (strong churn signal)."""

    def fit(self, X, y=None):  # noqa: ANN001, ANN202
        return self

    def transform(self, X):  # noqa: ANN001, ANN202
        X = X.copy()
        X["avg_charge_per_month"] = (
            X["total_charges"] / X["tenure_months"].clip(lower=1)
        )
        # High monthly charges with low tenure => price-sensitive churn risk.
        X["high_spend_short_tenure"] = (
            (X["monthly_charges"] > X["monthly_charges"].median())
            & (X["tenure_months"] < 12)
        ).astype(int)
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group infrequent categorical levels into 'other' to reduce sparsity."""

    def __init__(self, min_freq: float = 0.02, columns: list | None = None):
        self.min_freq = min_freq
        self.columns = columns or []
        self._keep: dict = {}

    def fit(self, X, y=None):  # noqa: ANN001, ANN202
        for col in self.columns:
            freq = X[col].value_counts(normalize=True)
            self._keep[col] = set(freq[freq >= self.min_freq].index)
        return self

    def transform(self, X):  # noqa: ANN001, ANN202
        X = X.copy()
        for col in self.columns:
            keep = self._keep[col]
            X[col] = np.where(X[col].isin(keep), X[col], "other")
        return X
