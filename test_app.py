"""API tests - run with: pytest -v"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

NORMAL = {"V1":-1.36,"V2":-0.073,"V3":2.536,"V4":1.378,"V5":-0.338,"V6":0.462,"V7":0.24,"V8":0.099,"V9":0.364,"V10":0.091,"V11":-0.552,"V12":-0.618,"V13":-0.991,"V14":-0.311,"V15":1.468,"V16":-0.47,"V17":0.208,"V18":0.026,"V19":0.404,"V20":0.251,"V21":-0.018,"V22":0.278,"V23":-0.11,"V24":0.067,"V25":0.129,"V26":-0.189,"V27":0.134,"V28":-0.021,"Amount":149.62}
FRAUD = {"V1":-2.312,"V2":1.952,"V3":-1.61,"V4":3.998,"V5":-0.522,"V6":-1.427,"V7":-2.537,"V8":1.392,"V9":-2.77,"V10":-2.772,"V11":3.202,"V12":-2.9,"V13":-0.595,"V14":-4.289,"V15":0.39,"V16":-1.141,"V17":-2.83,"V18":-0.017,"V19":0.417,"V20":0.127,"V21":0.517,"V22":-0.035,"V23":-0.465,"V24":0.32,"V25":0.045,"V26":0.178,"V27":0.261,"V28":-0.143,"Amount":123.45}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_normal():
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    d = r.json()
    assert d["prediction"] == "Normal"
    assert 0 <= d["fraud_probability"] <= 1


def test_predict_fraud():
    r = client.post("/predict", json=FRAUD)
    assert r.status_code == 200
    d = r.json()
    assert d["prediction"] == "Fraud"
    assert d["fraud_probability"] > 0.5
    assert len(d["explanation"]) >= 3  # SHAP explanation present


def test_predict_missing_field():
    bad = {k: v for k, v in NORMAL.items() if k != "V14"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
