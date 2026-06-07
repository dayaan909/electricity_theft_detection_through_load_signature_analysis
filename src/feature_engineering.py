"""
Feature engineering for electricity consumption profiles.
Derives behavioral indicators used for theft detection.
"""

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SEASONAL_FACTOR_JK = {
    "Jan": 1.45, "Feb": 1.40, "Mar": 1.10, "Apr": 0.95,
    "May": 0.85, "Jun": 0.80, "Jul": 0.78, "Aug": 0.82,
    "Sep": 0.90, "Oct": 1.00, "Nov": 1.25, "Dec": 1.40,
}

TARIFF_RATE = {
    "Residential": 4.50, "Commercial": 7.20,
    "Industrial": 6.80, "Agricultural": 2.50,
}


def compute_consumption_features(monthly_consumption: list, consumer_type: str = "Residential") -> dict:
    arr = np.array(monthly_consumption, dtype=float)
    tariff = TARIFF_RATE.get(consumer_type, 4.50)

    avg = float(np.mean(arr))
    std = float(np.std(arr))
    mx = float(np.max(arr))
    mn = float(np.min(arr))
    rng = mx - mn
    cv = std / avg if avg > 0 else 0

    winter_idx = [0, 1, 10, 11]
    summer_idx = [4, 5, 6]
    winter_avg = float(np.mean([arr[i] for i in winter_idx]))
    summer_avg = float(np.mean([arr[i] for i in summer_idx]))
    winter_summer_ratio = winter_avg / summer_avg if summer_avg > 0 else 1.0

    near_zero = int(np.sum(arr < 20))

    mom_changes = []
    for i in range(1, len(arr)):
        if arr[i - 1] > 0:
            mom_changes.append(abs(arr[i] - arr[i - 1]) / arr[i - 1] * 100)
        else:
            mom_changes.append(0.0)
    avg_mom = float(np.mean(mom_changes)) if mom_changes else 0.0
    max_mom = float(np.max(mom_changes)) if mom_changes else 0.0

    total_kwh = float(np.sum(arr))
    total_bill = total_kwh * tariff

    # Anomaly score (rule-based heuristic for quick estimation)
    anomaly_score = 0
    if cv > 0.6:
        anomaly_score += 25
    if near_zero > 2:
        anomaly_score += 30
    if max_mom > 70:
        anomaly_score += 20
    if winter_summer_ratio < 0.5 or winter_summer_ratio > 5.0:
        anomaly_score += 15
    if avg < 30:
        anomaly_score += 10

    return {
        "avg_monthly_consumption_kwh": round(avg, 2),
        "std_consumption": round(std, 2),
        "max_consumption_kwh": round(mx, 2),
        "min_consumption_kwh": round(mn, 2),
        "consumption_range_kwh": round(rng, 2),
        "coefficient_of_variation": round(cv, 4),
        "winter_avg_kwh": round(winter_avg, 2),
        "summer_avg_kwh": round(summer_avg, 2),
        "winter_summer_ratio": round(winter_summer_ratio, 4),
        "near_zero_months": near_zero,
        "avg_mom_change_pct": round(avg_mom, 2),
        "max_mom_change_pct": round(max_mom, 2),
        "total_annual_consumption_kwh": round(total_kwh, 2),
        "total_annual_bill_inr": round(total_bill, 2),
        "heuristic_anomaly_score": min(100, anomaly_score),
    }


def build_input_features(form_data: dict) -> dict:
    """
    Convert user form data into the feature dict required for model inference.
    form_data keys: consumer_type, sanctioned_load_kw, connected_load_kw,
                    years_as_consumer, payment_delay_avg_days, meter_status,
                    monthly_consumption (list of 12 floats)
    """
    monthly = form_data.get("monthly_consumption", [100] * 12)
    consumer_type = form_data.get("consumer_type", "Residential")

    features = compute_consumption_features(monthly, consumer_type)

    meter_status_map = {
        "Functioning": 0, "Slow Running": 1,
        "Tampered": 2, "Bypassed": 3, "Reversed": 4
    }
    consumer_type_map = {
        "Residential": 0, "Commercial": 1, "Industrial": 2, "Agricultural": 3
    }

    combined = {
        **features,
        "sanctioned_load_kw": float(form_data.get("sanctioned_load_kw", 5.0)),
        "connected_load_kw": float(form_data.get("connected_load_kw", 5.0)),
        "years_as_consumer": int(form_data.get("years_as_consumer", 5)),
        "payment_delay_avg_days": int(form_data.get("payment_delay_avg_days", 10)),
        "meter_status_enc": meter_status_map.get(
            form_data.get("meter_status", "Functioning"), 0
        ),
        "consumer_type_enc": consumer_type_map.get(consumer_type, 0),
        "district_enc": int(form_data.get("district_enc", 10)),
        "division_enc": int(form_data.get("division_enc", 0)),
    }

    # Add monthly columns
    for i, m in enumerate(MONTHS):
        combined[f"consumption_{m}_kwh"] = float(monthly[i]) if i < len(monthly) else 100.0

    return combined
