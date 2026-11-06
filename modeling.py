from __future__ import annotations

"""Transparent forecasting and environmental scenario calculations for synthetic ETC2-style GPS data."""

import numpy as np
import pandas as pd


def build_time_series(traffic: pd.DataFrame, frequency: str = "15min") -> pd.DataFrame:
    """Aggregate link probe observations to network-wide time bins."""
    if traffic.empty or "time_bin" not in traffic:
        return pd.DataFrame(columns=["time_bin", "probe_records", "mean_speed_kmh"])
    frame = traffic.copy()
    frame["time_bin"] = pd.to_datetime(frame["time_bin"], errors="coerce")
    frame = frame.dropna(subset=["time_bin"])
    if frame.empty:
        return pd.DataFrame(columns=["time_bin", "probe_records", "mean_speed_kmh"])
    frame["sample_count"] = pd.to_numeric(frame.get("sample_count", 0), errors="coerce").fillna(0)
    frame["mean_sample_speed_kmh"] = pd.to_numeric(frame.get("mean_sample_speed_kmh", np.nan), errors="coerce")
    frame["speed_weight"] = frame["sample_count"] * frame["mean_sample_speed_kmh"]
    grouped = frame.set_index("time_bin").resample(frequency).agg(
        probe_records=("sample_count", "sum"), speed_weight=("speed_weight", "sum"),
        speed_weight_count=("sample_count", "sum"))
    grouped["mean_speed_kmh"] = grouped["speed_weight"] / grouped["speed_weight_count"].replace(0, np.nan)
    return grouped[["probe_records", "mean_speed_kmh"]].reset_index()


def _feature_frame(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame({"value": series.astype(float)})
    for lag in (1, 2, 4, 96):
        frame[f"lag_{lag}"] = frame["value"].shift(lag)
    idx = frame.index
    frame["hour"] = idx.hour
    frame["weekday"] = idx.dayofweek
    frame["minute_bin"] = idx.hour * 4 + idx.minute // 15
    valid = frame.dropna()
    return valid.drop(columns="value"), valid["value"]


def evaluate_forecasts(series: pd.Series, target: str) -> tuple[pd.DataFrame, dict]:
    """Chronological holdout: compare persistence, daily-naive, and optional RF."""
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(s) < 24:
        return pd.DataFrame(), {"message": "At least 24 observed time bins are needed."}
    split = max(12, int(len(s) * .8))
    models: dict[str, list[float]] = {"Persistence baseline": [], "Previous-day baseline": []}
    actual: list[float] = []
    train = s.iloc[:split]
    test = s.iloc[split:]
    for stamp, observed in test.items():
        pos = s.index.get_loc(stamp)
        prev = float(s.iloc[pos - 1]) if pos > 0 else float(train.median())
        daily = float(s.iloc[pos - 96]) if pos >= 96 else prev
        actual.append(float(observed))
        models["Persistence baseline"].append(prev)
        models["Previous-day baseline"].append(daily)
    try:
        from sklearn.ensemble import RandomForestRegressor
        features, labels = _feature_frame(s)
        train_x = features.loc[features.index < test.index[0]]
        train_y = labels.loc[train_x.index]
        test_x = features.loc[features.index.isin(test.index)]
        if len(train_x) >= 30 and not test_x.empty:
            rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=-1)
            rf.fit(train_x, train_y)
            pred = pd.Series(rf.predict(test_x), index=test_x.index).reindex(test.index)
            models["Random forest"] = pred.fillna(pd.Series(models["Persistence baseline"], index=test.index)).tolist()
    except ImportError:
        pass
    rows = []
    y = np.asarray(actual, dtype=float)
    for name, predictions in models.items():
        p = np.asarray(predictions, dtype=float)
        err = p - y
        denom = np.maximum(np.abs(y), 1.0)
        rows.append({"model": name, "MAE": float(np.mean(np.abs(err))),
                     "RMSE": float(np.sqrt(np.mean(err ** 2))),
                     "sMAPE_percent": float(np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(p) + 1e-6)) * 100)})
    return pd.DataFrame(rows), {"train_bins": len(train), "test_bins": len(test),
                                "holdout_start": str(test.index.min()), "target": target}


def forecast_next(series: pd.Series, horizon_bins: int = 4, target: str = "") -> pd.DataFrame:
    """Persistence forecast (explicitly a baseline), recursively extending the series."""
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["time_bin", "forecast"])
    freq = pd.infer_freq(s.index) or "15min"
    future = pd.date_range(s.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon_bins, freq=freq)
    value = float(s.iloc[-1])
    return pd.DataFrame({"time_bin": future, "forecast": [max(0, value)] * horizon_bins})


def estimate_environment(links: pd.DataFrame, fuel_l_per_100km: float,
                         co2_kg_per_litre: float, idle_l_per_hour: float,
                         speed_threshold_kmh: float = 5.0) -> pd.DataFrame:
    """Apply user-selected average fuel factors; this is scenario arithmetic, not measured emissions."""
    if links.empty:
        return pd.DataFrame()
    df = links.copy()
    km = pd.to_numeric(df.get("gps_vehicle_km", 0), errors="coerce").fillna(0).clip(lower=0)
    speed = pd.to_numeric(df.get("mean_recorded_speed_kmh", np.nan), errors="coerce")
    near_stop_minutes = pd.to_numeric(df.get("near_stop_interval_minutes", 0), errors="coerce").fillna(0).clip(lower=0)
    idle_hours = near_stop_minutes / 60
    df["estimated_fuel_l"] = km * fuel_l_per_100km / 100 + idle_hours * idle_l_per_hour
    df["estimated_co2_kg"] = df["estimated_fuel_l"] * co2_kg_per_litre
    df["near_stop_interval_minutes"] = near_stop_minutes
    df["near_stop_hours_proxy"] = idle_hours
    df["speed_band"] = pd.cut(speed, [-np.inf, 15, 30, 50, np.inf], labels=["<15", "15-30", "30-50", ">50"])
    return df


def scenario_comparison(total_km: float, fuel_l_per_100km: float,
                        co2_kg_per_litre: float, detour_share: float,
                        speed_gain_percent: float, diverted_local_share: float) -> pd.DataFrame:
    """Illustrative route-policy scenarios. Changes are explicit sliders, not inferred behavior."""
    base_km = max(0, float(total_km))
    specs = [
        ("Observed synthetic route mix", 0.0, 0.0),
        ("Traffic-aware what-if", float(speed_gain_percent) / 100, 0.0),
        ("Residential-avoiding what-if", 0.0, float(detour_share)),
    ]
    rows = []
    for name, speed_gain, detour in specs:
        distance = base_km * (1 + detour)
        # Holding average speed uplift constant is a sensitivity scenario, not rerouting simulation.
        time_h = distance / max(1.0, 38.5 * (1 + speed_gain))
        fuel = distance * fuel_l_per_100km / 100
        rows.append({"scenario": name, "distance_vehicle_km": distance,
                     "relative_speed_gain_percent": speed_gain * 100,
                     "local_road_exposure_shift_percent": diverted_local_share if "Residential" in name else 0,
                     "illustrative_hours": time_h, "estimated_fuel_l": fuel,
                     "estimated_co2_kg": fuel * co2_kg_per_litre})
    return pd.DataFrame(rows)




