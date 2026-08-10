# 🕵️ Credit Card Fraud Detection API

Detect fraudulent credit card transactions with **Random Forest + SMOTE**, explained with **SHAP values** — deployed as a production FastAPI service with an interactive demo.

![tech](https://img.shields.io/badge/ML-RandomForest%2BSMOTE-0f766e) ![api](https://img.shields.io/badge/API-FastAPI-009688) ![shap](https://img.shields.io/badge/Explainability-SHAP-6d28d9)

## Why this project matters

Real-world fraud data is **insanely imbalanced**: only **0.173%** of 284,807 transactions are fraud. A model that says "everything is normal" gets 99.8% accuracy and is completely useless. This project tackles the *real* problem:

- **Recall matters most** — we must CATCH the frauds (missed fraud = money lost)
- **Precision matters too** — too many false alarms annoy customers (card blocked wrongly)
- **Explainability** — banks must explain *why* a transaction was flagged (regulation + trust)

## Model comparison (test set, 56,962 transactions, 98 frauds)

| Approach | ROC-AUC | Fraud Recall | Precision | F1 (fraud) |
|---|---|---|---|---|
| 1. Logistic Regression (balanced) | 0.971 | 81.6% | 83.3% | 0.825 |
| **2. Random Forest + SMOTE** ⭐ | **0.969** | **78.6%** | **96.2%** | **0.865** |
| 3. Isolation Forest (unsupervised) | 0.952 | 31.6% | 34.4% | 0.330 |
| 4. Autoencoder (unsupervised) | 0.940 | 38.8% | 38.8% | 0.388 |

**Winner: Random Forest + SMOTE** — 96.2% precision (only **3 false alarms** in 56k test transactions) and the best F1. Isolation Forest & Autoencoder show why unsupervised anomaly detection alone struggles on this data.

**SHAP top features** driving fraud predictions: V14, V12, V4, V10, V11.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/predict` | POST | 29 features → fraud probability + risk + **SHAP explanation** |
| `/` | GET | Interactive demo page |
| `/docs` | GET | Swagger UI |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -2.312, "V2": 1.952, "V3": -1.61, ..., "Amount": 123.45}'
```

### Example response

```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.98,
  "risk": "HIGH",
  "threshold": 0.80,
  "explanation": [
    {"feature": "V14", "shap": -0.42, "impact": "decreases fraud risk"},
    {"feature": "V12", "shap": 0.31,  "impact": "increases fraud risk"}
  ]
}
```

## Getting started

```bash
git clone <your-repo-url>
cd fraud-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Re)train the model - downloads creditcard.csv from TensorFlow's GCS mirror
python train_model.py

# Tests + run
pytest -v
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

## Project structure

```
├── app.py               # FastAPI: /predict + SHAP explanation + demo
├── train_model.py       # trains 4 approaches, evaluates, saves best + SHAP
├── test_app.py          # 4 API tests
├── models/              # trained artifacts (joblib)
├── static/index.html    # demo page (sample transactions + SHAP bars)
├── requirements.txt
├── Dockerfile
└── creditcard.csv       # 284,807 txns (gitignored - auto-download in train script)
```

## Key design decisions (interview gold 🎤)

1. **Fraud-focused metrics** — never accuracy on imbalanced data. We optimize F1 on the fraud class and report recall/precision.
2. **SMOTE + class_weight** — oversample the minority class so the tree sees enough fraud examples.
3. **Comparison vs unsupervised** — Isolation Forest & Autoencoder (reconstruction error) are honest baselines that show *why* supervised wins here.
4. **SHAP at inference** — the API computes per-transaction SHAP values so every prediction comes with a *reason* (V14 pushed it up, V12 down...).
5. **Threshold tuned by max-F1** — not a blind 0.5; the deploy threshold is where precision/recall balance best.

## Roadmap / stretch
- [ ] Streamlit-style dashboard with live transaction feed
- [ ] Precision@k evaluation (top-100 flagged reviews)
- [ ] A/B the threshold with cost model (false alarm ₹ vs missed fraud ₹)
