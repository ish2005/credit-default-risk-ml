# Credit Default Risk Prediction with Explainability and Production Deployment

An end-to-end machine learning project for predicting credit card default risk using **XGBoost, SHAP, MLflow, FastAPI, Docker, Google Cloud Run, and GitHub Actions**.

## Overview

This project predicts whether a credit card customer is likely to default on their next payment.

The workflow covers:

- Data preprocessing and feature engineering
- Handling class imbalance
- XGBoost model training
- Probability threshold optimization
- SHAP model explainability
- MLflow experiment tracking
- FastAPI REST API
- Docker containerization
- Google Cloud Run deployment
- Automated testing with GitHub Actions

## Dataset

The project uses the **UCI Credit Card Default dataset**, containing 30,000 customer records.

The target variable is:

- `0` → No default
- `1` → Default

The dataset is imbalanced, with approximately **77.9% non-default** and **22.1% default** cases.

## Feature Engineering

Additional behavioral features were created from repayment, billing, and payment history:

- `NUM_DELAYED` — number of delayed repayment periods
- `MAX_DELAY` — maximum repayment delay
- `RECENT_DELAY` — most recent repayment delay
- `AVG_BILL` — average bill amount
- `AVG_PAYMENT` — average payment amount

The final model uses a stratified **60/20/20 train/validation/test split**.

## Model

The final model is an **XGBoost classifier**.

The classification threshold was optimized on the validation set and selected as **0.60** based on F1 score.

### Final Test Results

| Metric | Score |
|---|---:|
| ROC-AUC | **0.777** |
| PR-AUC | **0.553** |
| Precision | **0.539** |
| Recall | **0.536** |
| F1 Score | **0.537** |
| Accuracy | **0.800** |

The model correctly identified **711 of 1,327 default cases** in the held-out test set.

## Explainability

**SHAP** was used to understand both global feature importance and individual predictions.

The most influential features included:

`NUM_DELAYED`, `MAX_DELAY`, `RECENT_DELAY`, `LIMIT_BAL`, `BILL_AMT1`, `PAY_AMT2`, `PAY_AMT1`, and `AVG_BILL`.

## MLOps & Deployment

**MLflow** was used for experiment tracking and model artifact management.

The trained model was packaged with **Skops** and served through a **FastAPI** REST API.

Deployment architecture:

```text
XGBoost Model
      ↓
   FastAPI
      ↓
    Docker
      ↓
Artifact Registry
      ↓
 Google Cloud Run
```

### Live API

**Swagger documentation:**  
https://credit-default-api-njyrsvkz7q-el.a.run.app/docs

### Example Response

```json
{
  "default_probability": 0.9013,
  "prediction": 1
}
```

`prediction = 1` indicates predicted default.

## Testing

The project includes automated API tests using **Pytest**.

GitHub Actions runs the test suite automatically on pushes and pull requests.

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · MLflow · Skops · FastAPI · Docker · Google Cloud Run · GitHub Actions · Pytest**

## Project Structure

```text
credit-default-risk-ml/
├── api/
├── src/
├── models/
├── tests/
├── notebooks/
├── data/
├── .github/workflows/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Limitations

- The dataset is historical.
- Model performance is moderate and should not be interpreted as banking-grade production performance.
- The threshold was optimized for F1 rather than a specific financial cost function.
- SHAP explains model behavior but does not establish causality.
- Production monitoring and drift detection are outside the current scope.
