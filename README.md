# fintech-antifraud-ml

Production-grade antifraud ML system (Data Science + MLOps signals):
- data validation + quality report
- feature engineering
- training with model selection (logreg vs hgb) + optional calibration
- cost-based threshold optimization (FP/FN business cost)
- drift detection (PSI + KS)
- explainability (permutation importance)
- versioned artifacts (`artifacts/runs/<run_id>`) + `artifacts/latest`
- FastAPI scoring service with request_id logging + Prometheus metrics

## Requirements
- Windows 11
- Python 3.12
- Poetry

## Install
```powershell
poetry install
