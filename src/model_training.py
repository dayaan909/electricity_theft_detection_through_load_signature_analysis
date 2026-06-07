"""
Trains and persists ML models for electricity theft detection.
Models: Isolation Forest (unsupervised), Random Forest, XGBoost (supervised).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import load_data, prepare_train_test, scale_features, get_feature_matrix

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "jk_electricity_consumers.csv")


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train):
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    return xgb_model


def train_isolation_forest(X_train):
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.22,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)
    return iso


def evaluate_supervised(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_isolation_forest(model, X_test, y_test) -> dict:
    # IsolationForest: -1 = anomaly, 1 = normal → map to 1/0
    preds_raw = model.predict(X_test)
    y_pred = np.where(preds_raw == -1, 1, 0)
    scores = -model.score_samples(X_test)  # higher = more anomalous

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, scores)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Isolation Forest Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")

    return {
        "model": "Isolation Forest",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }


def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    print("\nPreparing train/test split...")
    X_train, X_test, y_train, y_test = prepare_train_test(df, test_size=0.2)

    print("Scaling features...")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Feature names for scaler reference
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    results = []

    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train_sc, y_train)
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    joblib.dump(rf, rf_path)
    print(f"Random Forest saved to: {rf_path}")
    res_rf = evaluate_supervised(rf, X_test_sc, y_test, "Random Forest")
    results.append(res_rf)

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train_sc, y_train)
    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    joblib.dump(xgb_model, xgb_path)
    print(f"XGBoost saved to: {xgb_path}")
    res_xgb = evaluate_supervised(xgb_model, X_test_sc, y_test, "XGBoost")
    results.append(res_xgb)

    print("\nTraining Isolation Forest...")
    iso = train_isolation_forest(X_train_sc)
    iso_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    joblib.dump(iso, iso_path)
    print(f"Isolation Forest saved to: {iso_path}")
    res_iso = evaluate_isolation_forest(iso, X_test_sc, y_test)
    results.append(res_iso)

    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = os.path.join(MODELS_DIR, "model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "="*50)
    print("  TRAINING COMPLETE — Summary")
    print("="*50)
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    run_training()
