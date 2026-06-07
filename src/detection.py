"""
Inference module: loads trained models and runs theft detection on new inputs.
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_cache = {}


def _load(name: str):
    if name not in _cache:
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            _cache[name] = joblib.load(path)
        else:
            _cache[name] = None
    return _cache[name]


def models_available() -> bool:
    required = ["random_forest.pkl", "xgboost_model.pkl", "isolation_forest.pkl", "scaler.pkl"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def prepare_input(feature_dict: dict) -> np.ndarray:
    scaler = _load("scaler.pkl")
    feature_names = _load("feature_names.pkl")

    if scaler is None or feature_names is None:
        raise RuntimeError("Scaler or feature names not found. Train models first.")

    row = []
    for feat in feature_names:
        row.append(float(feature_dict.get(feat, 0.0)))

    X = np.array(row).reshape(1, -1)
    return scaler.transform(X), feature_names


def predict_all(feature_dict: dict) -> dict:
    """
    Runs all three models on the feature dict.
    Returns a results dict with per-model predictions and a combined verdict.
    """
    if not models_available():
        return {"error": "Models not trained yet. Please run training first."}

    X_scaled, _ = prepare_input(feature_dict)

    rf = _load("random_forest.pkl")
    xgb_model = _load("xgboost_model.pkl")
    iso = _load("isolation_forest.pkl")

    results = {}

    # Random Forest
    rf_pred = int(rf.predict(X_scaled)[0])
    rf_prob = float(rf.predict_proba(X_scaled)[0][1])
    results["random_forest"] = {
        "prediction": rf_pred,
        "label": "THEFT DETECTED" if rf_pred == 1 else "NORMAL",
        "confidence_pct": round(rf_prob * 100, 1),
    }

    # XGBoost
    xgb_pred = int(xgb_model.predict(X_scaled)[0])
    xgb_prob = float(xgb_model.predict_proba(X_scaled)[0][1])
    results["xgboost"] = {
        "prediction": xgb_pred,
        "label": "THEFT DETECTED" if xgb_pred == 1 else "NORMAL",
        "confidence_pct": round(xgb_prob * 100, 1),
    }

    # Isolation Forest
    iso_raw = int(iso.predict(X_scaled)[0])
    iso_pred = 1 if iso_raw == -1 else 0
    iso_score = float(-iso.score_samples(X_scaled)[0])
    results["isolation_forest"] = {
        "prediction": iso_pred,
        "label": "ANOMALY DETECTED" if iso_pred == 1 else "NORMAL",
        "anomaly_score": round(iso_score, 4),
    }

    # Ensemble verdict: majority vote among supervised models
    votes = rf_pred + xgb_pred + iso_pred
    avg_prob = (rf_prob + xgb_prob) / 2
    verdict = "THEFT LIKELY" if votes >= 2 else "NORMAL"
    risk_level = (
        "HIGH" if avg_prob >= 0.70 else
        "MEDIUM" if avg_prob >= 0.40 else
        "LOW"
    )

    results["ensemble"] = {
        "verdict": verdict,
        "votes_for_theft": votes,
        "average_probability_pct": round(avg_prob * 100, 1),
        "risk_level": risk_level,
    }

    return results
