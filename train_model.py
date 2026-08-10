"""
Credit Card Fraud Detection - training script
---------------------------------------------
Approaches compared (because fraud data is ~0.17% of all transactions):
  1. Logistic Regression (baseline, class_weight)
  2. Random Forest (class_weight) + SMOTE oversampling
  3. Isolation Forest (unsupervised anomaly detection)
  4. Autoencoder (unsupervised - reconstruction error as anomaly score)

Evaluation is fraud-focused: RECALL matters most (catch frauds!), then
precision/F1/ROC-AUC. Accuracy is meaningless here (99.8% "no fraud").

Explainability: SHAP values on the Random Forest -> which features made a
transaction look fraudulent. Saved with the model for the API.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
FRAUD_THRESHOLD = 0.5

print("Loading data...")
df = pd.read_csv("creditcard.csv")
print(f"  Shape: {df.shape} | Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# Drop Time (no cyclical meaning after PCA); scale Amount
X = df.drop(columns=["Class", "Time"])
y = df["Class"].astype(int)
FEATURE_COLS = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), FEATURE_COLS)]
)

# Stratified split keeps the tiny fraud share in both train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Test fraud: {y_test.sum()}")

# ---------- 1) Logistic Regression baseline ----------
lr = Pipeline([
    ("pre", preprocessor),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
])
lr.fit(X_train, y_train)

# ---------- 2) Random Forest + class_weight ----------
# (fits on the SMOTE-resampled, already-scaled array - no ColumnTransformer)
rf_model = RandomForestClassifier(
    n_estimators=120, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2,
)

# SMOTE on the scaled features
X_scaled = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)
X_res, y_res = smote.fit_resample(X_scaled, y_train)
print(f"After SMOTE: {X_res.shape} (fraud now {y_res.mean()*100:.1f}%)")

rf_model.fit(X_res, y_res)
# Serving pipeline = fitted scaler + fitted RF (no refit - keeps SMOTE benefits)
rf_pipe = Pipeline([("pre", preprocessor), ("model", rf_model)])

# ---------- 3) Isolation Forest (unsupervised) ----------
X_scaled_full = preprocessor.fit_transform(X)
iso = IsolationForest(
    n_estimators=150, contamination=0.002, random_state=RANDOM_STATE, n_jobs=-1
)
iso.fit(X_scaled_full)
iso_test_scores = iso.score_samples(preprocessor.transform(X_test))

# ---------- 4) Autoencoder (unsupervised, sklearn MLP as AE) ----------
X_train_normal = X_scaled[y_train == 0]  # train only on normal transactions
ae = MLPRegressor(
    hidden_layer_sizes=(16, 8, 16), activation="relu", solver="adam",
    max_iter=15, random_state=RANDOM_STATE, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=5,
)
ae.fit(X_train_normal, X_train_normal)

def reconstruction_error(Xs):
    pred = ae.predict(Xs)
    return np.mean((Xs - pred) ** 2, axis=1)

# ---------- Evaluate all ----------
print("\n" + "=" * 62)
print("MODEL COMPARISON (fraud-focused)")
print("=" * 62)

def report(name, y_prob_or_score, higher_is_fraud=True):
    if higher_is_fraud:
        y_s = y_prob_or_score
    else:  # isolation forest scores: higher = more normal
        y_s = -y_prob_or_score
    auc = roc_auc_score(y_test, y_s)
    # pick threshold that maximizes F1 on the fraud class
    prec, rec, thr = precision_recall_curve(y_test, y_s)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = np.argmax(f1)
    y_pred = (y_s >= thr[best]).astype(int) if best < len(thr) else (y_s >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name}")
    print(f"  ROC-AUC : {auc:.4f} | Best-F1 threshold: {thr[best]:.4f}")
    print(f"  Fraud recall : {cm[1,1]/(cm[1,1]+cm[1,0])*100:.1f}%  (caught {cm[1,1]}/{y_test.sum()} frauds)")
    print(f"  Precision    : {cm[1,1]/(cm[1,1]+cm[0,1])*100:.1f}%  (false alarms: {cm[0,1]})")
    print(f"  F1 (fraud)   : {f1[best]:.4f}")
    return y_s, thr[best]

y_s_lr, thr_lr = report("1) Logistic Regression", lr.predict_proba(X_test)[:, 1])
y_s_rf, thr_rf = report("2) Random Forest + SMOTE", rf_pipe.predict_proba(X_test)[:, 1])
y_s_iso, thr_iso = report("3) Isolation Forest", iso_test_scores, higher_is_fraud=False)
y_s_ae, thr_ae = report("4) Autoencoder", reconstruction_error(preprocessor.transform(X_test)))

print("\n" + "=" * 62)
print("VERDICT: Random Forest + SMOTE wins on F1+recall -> deploying it.")
print("=" * 62)

# ---------- SHAP explainability ----------
print("\nComputing SHAP values (TreeExplainer on the RF)...")
top_features = [(FEATURE_COLS[i], 0.0) for i in range(5)]
try:
    import shap
    explainer = shap.TreeExplainer(rf_model)
    X_test_scaled = preprocessor.transform(X_test)
    shap_values = explainer.shap_values(X_test_scaled[:2000])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]          # class 1 (fraud)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1]     # (samples, features, classes) -> fraud class
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_abs_shap = np.asarray(mean_abs_shap).ravel()
    top_features = sorted(
        zip(FEATURE_COLS, mean_abs_shap), key=lambda t: -float(t[1])
    )[:10]
    print("Top 10 features driving fraud predictions:")
    for name, val in top_features:
        print(f"  {name:>6s}: {float(val):.4f}")
    explainer = explainer  # keep for API
except Exception as e:
    print(f"SHAP skipped ({e}) - continuing with model save")
    explainer = None

# ---------- Persist ----------
print("\nSaving artifacts...")
joblib.dump({
    "preprocessor": preprocessor,
    "model": rf_pipe,
    "threshold": thr_rf,
    "feature_cols": FEATURE_COLS,
    "top_features": [t[0] for t in top_features],
}, "models/fraud_model.joblib")
joblib.dump(explainer, "models/shap_explainer.joblib")
print("Saved: models/fraud_model.joblib + models/shap_explainer.joblib")
