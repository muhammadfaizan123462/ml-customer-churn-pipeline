# Customer Churn Prediction — End-to-End ML Pipeline

An end-to-end machine learning project that predicts telecom customer churn:
synthetic data generation → custom feature engineering → tuned gradient
boosting → evaluation → a production-ready FastAPI inference service
(containerised with Docker).

## Project structure

```
ml-customer-churn-pipeline/
├── src/
│   ├── data.py        # reproducible synthetic dataset generator
│   ├── features.py    # custom sklearn transformers (tenure binning,
│   │                  #   charge ratios, rare-category grouping)
│   ├── train.py       # Pipeline + RandomizedSearchCV tuning, saves artifacts
│   ├── evaluate.py    # metrics, ROC/PR curves, permutation importance
│   └── api.py         # FastAPI inference service with pydantic validation
├── tests/
│   └── test_smoke.py  # fast end-to-end smoke tests
├── Dockerfile         # containerised API
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data
python src/data.py --n-samples 5000

# 2. Train (tunes HistGradientBoosting with 5-fold CV)
python src/train.py --n-iter 20

# 3. Evaluate
python src/evaluate.py

# 4. Serve the API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## API

- `GET /health` — service & model status
- `POST /predict` — JSON customer features →
  `{churn_probability, churn_predicted, risk_band}`

## Results (measured on the synthetic dataset, 5-fold CV)

| Metric | Test set |
|---|---|
| ROC AUC | 0.889 |
| Accuracy | 0.870 |
| Precision | 0.838 |
| Recall | 0.671 |
| F1 | 0.745 |

Top features by permutation importance: `data_usage_gb`, `tenure_months`,
`support_calls` — sensible drivers of churn.

## Highlights

- **Custom transformers** (`TenureBinner`, `ChargeRatioAdder`,
  `RareCategoryGrouper`) composed inside a single sklearn `Pipeline`,
  so preprocessing is identical at train and serve time — no skew.
- **RandomizedSearchCV** over learning rate, tree depth, leaf size and
  L2 regularisation with stratified 5-fold CV, optimising ROC AUC.
- **Experiment artifacts**: `artifacts/model.joblib`,
  `artifacts/metrics.json`, `artifacts/params.json` — every run is
  reproducible from a fixed random seed.
- **Serve-ready**: pydantic-validated FastAPI service with a Dockerfile.
