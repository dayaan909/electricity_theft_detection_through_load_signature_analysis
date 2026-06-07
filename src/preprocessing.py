"""
Data preprocessing pipeline for electricity theft detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

MONTH_COLS = [
    "consumption_Jan_kwh", "consumption_Feb_kwh", "consumption_Mar_kwh",
    "consumption_Apr_kwh", "consumption_May_kwh", "consumption_Jun_kwh",
    "consumption_Jul_kwh", "consumption_Aug_kwh", "consumption_Sep_kwh",
    "consumption_Oct_kwh", "consumption_Nov_kwh", "consumption_Dec_kwh",
]

FEATURE_COLS = [
    "avg_monthly_consumption_kwh", "std_consumption", "max_consumption_kwh",
    "min_consumption_kwh", "consumption_range_kwh", "coefficient_of_variation",
    "winter_avg_kwh", "summer_avg_kwh", "winter_summer_ratio",
    "near_zero_months", "avg_mom_change_pct", "max_mom_change_pct",
    "total_annual_consumption_kwh", "total_annual_bill_inr",
    "sanctioned_load_kw", "connected_load_kw",
    "years_as_consumer", "payment_delay_avg_days",
] + MONTH_COLS


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = ["consumer_type", "district", "division", "meter_status"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_categoricals(df)
    extra_enc = ["consumer_type_enc", "district_enc", "division_enc", "meter_status_enc"]
    available = [c for c in FEATURE_COLS + extra_enc if c in df.columns]
    return df[available]


def scale_features(X_train, X_test=None, scaler_path: str = None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.2):
    X = get_feature_matrix(df)
    y = df["theft_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_single_input(input_dict: dict, scaler) -> np.ndarray:
    """Scale a single consumer input dict for inference."""
    df_single = pd.DataFrame([input_dict])
    feature_names = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None
    if feature_names is not None:
        for col in feature_names:
            if col not in df_single.columns:
                df_single[col] = 0
        df_single = df_single[feature_names]
    return scaler.transform(df_single)
