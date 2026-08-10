"""
Credit Card Fraud Detection API
--------------------------------
Endpoints:
  POST /predict  -> fraud probability + risk + SHAP explanation
  GET  /         -> interactive demo page

Pipeline: scale (StandardScaler) -> RandomForest(+SMOTE) -> probability
Explainability: SHAP TreeExplainer on the RF -> top contributing features.
"""
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODEL_PATH = "models/fraud_model.joblib"
data = joblib.load(MODEL_PATH)
preprocessor = data["preprocessor"]
model = data["model"]
THRESHOLD = data["threshold"]
FEATURE_COLS = data["feature_cols"]

# SHAP explainer built from the RF (fast for trees)
import shap
explainer = shap.TreeExplainer(model.named_steps["model"] if hasattr(model, "named_steps") else model)
model_core = model.named_steps["model"] if hasattr(model, "named_steps") else model

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")


class Transaction(BaseModel):
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., description="Transaction amount (USD)")


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


@app.post("/predict")
def predict(tx: Transaction):
    row = pd.DataFrame([tx.model_dump()])[FEATURE_COLS]
    X = preprocessor.transform(row)

    prob_fraud = float(model_core.predict_proba(X)[0, 1])
    pred = "Fraud" if prob_fraud >= THRESHOLD else "Normal"
    risk = "HIGH" if prob_fraud >= 0.8 else ("MEDIUM" if prob_fraud >= 0.5 else "LOW")

    # SHAP explanation: which features drove this prediction?
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values).ravel()
    contrib = sorted(zip(FEATURE_COLS, shap_values), key=lambda t: -abs(float(t[1])))[:5]
    explanation = [
        {"feature": f, "shap": round(float(v), 4), "impact": "increases fraud risk" if v > 0 else "decreases fraud risk"}
        for f, v in contrib
    ]

    return {
        "prediction": pred,
        "fraud_probability": round(prob_fraud, 4),
        "risk": risk,
        "threshold": THRESHOLD,
        "explanation": explanation,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
