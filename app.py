import io
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.set_page_config(page_title="Electricity Demand Forecasting", page_icon="⚡", layout="wide")


MODEL_FILES = {
    "Random Forest": ["random_forest.pkl", "random_forest_model.pkl"],
    "XGBoost": ["xgboost.pkl", "xgboost_model.pkl", "xgboost_forecasting_model.pkl"],
    "Linear Regression": ["linear_regression.pkl", "linear_regression_model.pkl"],
}
FEATURE_FILE_CANDIDATES = ["feature_columns.pkl", "xgboost_feature_columns.pkl"]
SCALER_FILE_CANDIDATES = ["scaler.pkl", "standard_scaler.pkl"]


# -----------------------------
# Helpers
# -----------------------------
def _find_folder(candidates: List[str]) -> Path:
    base = Path(__file__).resolve().parent
    for c in candidates:
        p = base / c
        if p.exists() and p.is_dir():
            return p
    return base / candidates[0]


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def _load_pickle(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data_dir = _find_folder(["data", "Data"])
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV found in data/Data folder.")

    file_path = csv_files[0]
    return load_and_clean_data(str(file_path))


@st.cache_data(show_spinner=False)
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    # 1) Load raw data and normalize common missing markers.
    df = pd.read_csv(file_path, low_memory=False, na_values=["?", "NA", "N/A", "na", ""])
    if df.empty:
        st.error("Loaded dataset is empty.")
        return pd.DataFrame()

    # 2) Create Datetime and keep lowercase alias for existing app logic.
    if {"Date", "Time"}.issubset(df.columns):
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            dayfirst=True,
            errors="coerce",
        )
    else:
        dt_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if dt_candidates:
            df["Datetime"] = pd.to_datetime(df[dt_candidates[0]], errors="coerce")
        else:
            st.error("Could not infer date/time columns to create Datetime.")
            return pd.DataFrame()

    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    if df.empty:
        st.error("No valid rows after Datetime parsing.")
        return pd.DataFrame()
    df["datetime"] = df["Datetime"]

    # 3) Convert required columns to numeric.
    numeric_candidates = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) Missing value strategy.
    if "Global_active_power" not in df.columns:
        st.error("Required target column 'Global_active_power' is missing.")
        return pd.DataFrame()

    df = df.dropna(subset=["Global_active_power"]).reset_index(drop=True)
    if df.empty:
        st.error("No rows left after dropping missing Global_active_power.")
        return pd.DataFrame()

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = df[col].ffill()
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

    # 5) Feature engineering for analytics and ML.
    df["Hour"] = df["Datetime"].dt.hour
    df["Day"] = df["Datetime"].dt.day
    df["Month"] = df["Datetime"].dt.month
    df["hour"] = df["Hour"]
    df["day"] = df["Day"]
    df["month"] = df["Month"]
    df["day_of_week"] = df["Datetime"].dt.dayofweek

    target = "Global_active_power"
    df["rolling_mean_5"] = df[target].rolling(window=5, min_periods=1).mean()
    df["rolling_mean_10"] = df[target].rolling(window=10, min_periods=1).mean()
    df["lag_1"] = df[target].shift(1)
    df["lag_24"] = df[target].shift(24)

    # Keep compatibility with model artifacts already used in this dashboard.
    df["lag_2"] = df[target].shift(2)
    df["lag_3"] = df[target].shift(3)
    df["lag_7"] = df[target].shift(7)
    df["lag_60"] = df[target].shift(60)
    df["lag_1440"] = df[target].shift(1440)
    df["rolling_mean_3"] = df[target].rolling(window=3, min_periods=1).mean()
    df["rolling_mean_60"] = df[target].rolling(window=60, min_periods=1).mean()
    df["rolling_std_5"] = df[target].rolling(window=5, min_periods=1).std().fillna(0.0)
    df["rolling_std_60"] = df[target].rolling(window=60, min_periods=1).std().fillna(0.0)

    # 6) Final cleanup: enforce float for numeric features and remove any remaining NaNs.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Streamlit integration debug snapshots.
    st.write(df.dtypes)
    st.write(df.head())

    return df


def detect_target_column(df: pd.DataFrame) -> str:
    preferred = [
        "Global_active_power",
        "consumption",
        "electricity_consumption",
        "demand",
        "load",
        "target",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric target-like column found.")
    return numeric_cols[0]


def infer_optional_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {"temperature": None, "region": None}
    for c in df.columns:
        cl = c.lower()
        if mapping["temperature"] is None and "temp" in cl:
            mapping["temperature"] = c
        if mapping["region"] is None and any(k in cl for k in ["region", "zone", "area", "city"]):
            mapping["region"] = c
    return mapping


@st.cache_resource(show_spinner=False)
def load_models_and_meta() -> Tuple[Dict[str, object], List[str], Optional[object]]:
    model_dir = _find_folder(["models", "Models"])
    loaded: Dict[str, object] = {}

    for model_name, file_candidates in MODEL_FILES.items():
        for file_name in file_candidates:
            p = model_dir / file_name
            if p.exists():
                loaded[model_name] = _load_pickle(p)
                break

    feature_columns: List[str] = []
    for f in FEATURE_FILE_CANDIDATES:
        fp = model_dir / f
        if fp.exists():
            cols = _load_pickle(fp)
            if isinstance(cols, list):
                feature_columns = cols
                break

    scaler = None
    for s in SCALER_FILE_CANDIDATES:
        sp = model_dir / s
        if sp.exists():
            scaler = _load_pickle(sp)
            break

    return loaded, feature_columns, scaler


@st.cache_data(show_spinner=False)
def prepare_feature_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Guard against invalid inputs and keep behavior predictable for downstream models.
    if df is None or df.empty:
        st.error("Input dataframe is empty.")
        return pd.DataFrame()
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' is missing from dataframe.")
        return pd.DataFrame()

    work = df.copy(deep=True)

    # Ensure datetime exists and is valid.
    if "datetime" not in work.columns:
        st.error("Missing required 'datetime' column for feature engineering.")
        return pd.DataFrame()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if work.empty:
        st.error("No valid rows after parsing datetime.")
        return pd.DataFrame()

    # Create target and force numeric conversion so rolling windows never run on object dtype.
    work["target"] = pd.to_numeric(work[target_col], errors="coerce")
    st.caption(f"Target dtype after conversion: {work['target'].dtype}")
    st.caption(f"Target sample (first 5): {work['target'].head(5).tolist()}")

    # Validate target quality.
    if not pd.api.types.is_numeric_dtype(work["target"]):
        st.error("Target column must be numeric")
        return pd.DataFrame()

    # Remove unusable target rows first; interpolate remaining gaps for stable feature windows.
    work = work.dropna(subset=["target"]).reset_index(drop=True)
    if work.empty:
        st.error("Target column must be numeric")
        return pd.DataFrame()
    work["target"] = work["target"].interpolate(limit_direction="both").ffill().bfill()

    # Requested robust rolling and lag features.
    work["rolling_mean_3"] = work["target"].rolling(window=3, min_periods=1).mean()
    work["rolling_mean_5"] = work["target"].rolling(window=5, min_periods=1).mean()
    work["rolling_std_5"] = work["target"].rolling(window=5, min_periods=1).std().fillna(0.0)
    work["lag_1"] = work["target"].shift(1)
    work["lag_2"] = work["target"].shift(2)
    work["lag_7"] = work["target"].shift(7)

    # Keep additional legacy features for compatibility with existing trained artifacts.
    work["lag_3"] = work["target"].shift(3)
    work["lag_60"] = work["target"].shift(60)
    work["lag_1440"] = work["target"].shift(1440)
    work["rolling_mean_60"] = work["target"].rolling(window=60, min_periods=1).mean()
    work["rolling_std_60"] = work["target"].rolling(window=60, min_periods=1).std().fillna(0.0)

    # Calendar features.
    work["hour"] = work["datetime"].dt.hour
    work["day_of_week"] = work["datetime"].dt.dayofweek
    work["month"] = work["datetime"].dt.month
    work["day"] = work["datetime"].dt.day

    # Final cleaning for ML readiness.
    work = work.replace([np.inf, -np.inf], np.nan)
    feature_cols = [
        "target",
        "rolling_mean_3",
        "rolling_mean_5",
        "rolling_std_5",
        "lag_1",
        "lag_2",
        "lag_7",
        "lag_3",
        "lag_60",
        "lag_1440",
        "rolling_mean_60",
        "rolling_std_60",
        "hour",
        "day_of_week",
        "month",
        "day",
    ]
    existing_feature_cols = [c for c in feature_cols if c in work.columns]
    work[existing_feature_cols] = work[existing_feature_cols].ffill().bfill()
    work = work.dropna(subset=existing_feature_cols).reset_index(drop=True)

    return work


@st.cache_resource(show_spinner=False)
def build_fallback_random_forest(feature_df: pd.DataFrame, feature_columns: List[str]):
    needed = feature_columns if feature_columns else [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_60",
        "lag_1440",
        "rolling_mean_5",
        "rolling_mean_60",
        "rolling_std_60",
        "hour",
        "day_of_week",
        "month",
        "day",
    ]

    available = [c for c in needed if c in feature_df.columns]
    if len(available) < 4:
        return None

    sample_df = feature_df.tail(min(60000, len(feature_df))).copy()
    x = sample_df[available]
    y = sample_df["target"]

    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x, y)
    return rf


def style_app(dark_mode: bool) -> None:
    if dark_mode:
        bg = "#0f172a"
        card = "#111827"
        text = "#e5e7eb"
        accent = "#22c55e"
        accent2 = "#38bdf8"
    else:
        bg = "#f4fbff"
        card = "#ffffff"
        text = "#0b2d39"
        accent = "#16a34a"
        accent2 = "#0284c7"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background: radial-gradient(circle at top right, rgba(2,132,199,0.10), transparent 42%),
                            radial-gradient(circle at bottom left, rgba(22,163,74,0.12), transparent 40%),
                            {bg};
                color: {text};
            }}
            [data-testid='stSidebar'] {{
                background: linear-gradient(180deg, #052e45 0%, #064e3b 100%);
            }}
            [data-testid='stSidebar'] * {{
                color: #e6fffb !important;
            }}
            .energy-card {{
                border-radius: 14px;
                background: {card};
                padding: 16px;
                border: 1px solid rgba(2,132,199,0.18);
                box-shadow: 0 8px 24px rgba(2, 132, 199, 0.08);
                margin-bottom: 14px;
            }}
            .highlight {{
                padding: 14px 16px;
                border-left: 5px solid {accent};
                background: rgba(22, 163, 74, 0.12);
                border-radius: 10px;
                font-weight: 700;
            }}
            .small-note {{
                color: {accent2};
                font-size: 0.92rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_feature_row(
    selected_dt: pd.Timestamp,
    feature_df: pd.DataFrame,
    temperature_value: float,
    manual_lags: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    latest = feature_df.iloc[-1].copy()

    payload = {
        "lag_1": float(latest.get("lag_1", latest["target"])),
        "lag_2": float(latest.get("lag_2", latest["target"])),
        "lag_3": float(latest.get("lag_3", latest["target"])),
        "lag_60": float(latest.get("lag_60", latest["target"])),
        "lag_1440": float(latest.get("lag_1440", latest["target"])),
        "rolling_mean_5": float(latest.get("rolling_mean_5", latest["target"])),
        "rolling_mean_60": float(latest.get("rolling_mean_60", latest["target"])),
        "rolling_std_60": float(latest.get("rolling_std_60", 0.0)),
        "hour": int(selected_dt.hour),
        "day_of_week": int(selected_dt.dayofweek),
        "month": int(selected_dt.month),
        "day": int(selected_dt.day),
        "temperature": float(temperature_value),
    }

    if manual_lags:
        payload.update(manual_lags)

    return pd.DataFrame([payload])


def align_features(x: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    if not feature_columns:
        return x
    for c in feature_columns:
        if c not in x.columns:
            x[c] = 0.0
    return x[feature_columns]


def model_predict(model_name: str, model, x: pd.DataFrame, scaler: Optional[object]) -> np.ndarray:
    x_in = x.copy()
    if scaler is not None and hasattr(scaler, "transform") and model_name == "Linear Regression":
        scaler_any = cast(Any, scaler)
        x_in = pd.DataFrame(scaler_any.transform(x_in), columns=x_in.columns)
    pred = model.predict(x_in)
    return np.asarray(pred).reshape(-1)


def recursive_forecast(
    model_name: str,
    model,
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    scaler: Optional[object],
    horizon: int,
) -> pd.DataFrame:
    history = feature_df["target"].tail(2000).tolist()
    std60 = float(np.std(history[-60:])) if len(history) >= 60 else float(np.std(history))

    start_ts = feature_df["datetime"].iloc[-1]
    out_rows = []
    for step in range(1, horizon + 1):
        ts = start_ts + pd.Timedelta(minutes=step)

        if len(history) >= 60:
            rm5 = float(np.mean(history[-5:]))
            rm60 = float(np.mean(history[-60:]))
        else:
            rm5 = float(np.mean(history))
            rm60 = float(np.mean(history))

        row = pd.DataFrame(
            [
                {
                    "lag_1": float(history[-1]),
                    "lag_2": float(history[-2]) if len(history) > 1 else float(history[-1]),
                    "lag_3": float(history[-3]) if len(history) > 2 else float(history[-1]),
                    "lag_60": float(history[-60]) if len(history) >= 60 else float(history[-1]),
                    "lag_1440": float(history[-1440]) if len(history) >= 1440 else float(history[-1]),
                    "rolling_mean_5": rm5,
                    "rolling_mean_60": rm60,
                    "rolling_std_60": std60,
                    "hour": int(ts.hour),
                    "day_of_week": int(ts.dayofweek),
                    "month": int(ts.month),
                    "day": int(ts.day),
                }
            ]
        )

        row = align_features(row, feature_columns)
        y_hat = float(model_predict(model_name, model, row, scaler)[0])
        history.append(y_hat)
        out_rows.append({"datetime": ts, "predicted": y_hat})

    return pd.DataFrame(out_rows)


def evaluate_models(
    models: Dict[str, object],
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    scaler: Optional[object],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    test_size = min(1200, max(200, int(len(feature_df) * 0.2)))
    train_df = feature_df.iloc[:-test_size].copy()
    test_df = feature_df.iloc[-test_size:].copy()

    reports = []
    pred_tables: Dict[str, pd.DataFrame] = {}

    for name, model in models.items():
        used_cols = feature_columns if feature_columns else [c for c in test_df.columns if c not in ["datetime", "target"]]
        used_cols = [c for c in used_cols if c in test_df.columns]
        if not used_cols:
            continue

        x_test = test_df[used_cols]
        x_test = align_features(x_test, feature_columns)
        y_true = test_df["target"].to_numpy()

        try:
            y_pred = model_predict(name, model, x_test, scaler)
        except Exception:
            continue

        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        smape = _safe_smape(y_true, y_pred)
        reports.append({"Model": name, "MAE": mae, "RMSE": rmse, "SMAPE": smape})
        pred_tables[name] = pd.DataFrame(
            {
                "datetime": test_df["datetime"].values,
                "actual": y_true,
                "predicted": y_pred,
            }
        )

    if not reports:
        return pd.DataFrame(columns=["Model", "MAE", "RMSE", "SMAPE"]), {}

    metrics_df = pd.DataFrame(reports).sort_values("RMSE").reset_index(drop=True)
    return metrics_df, pred_tables


# -----------------------------
# Pages
# -----------------------------
def render_dashboard_overview(df: pd.DataFrame, target_col: str) -> None:
    st.title("⚡ Dashboard Overview")
    st.caption("High-level view of electricity demand behavior and trends.")

    total = float(df[target_col].sum())
    avg = float(df[target_col].mean())
    peak = float(df[target_col].max())
    minimum = float(df[target_col].min())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Electricity Consumption", f"{total:,.2f}")
    c2.metric("Average Consumption", f"{avg:,.3f}")
    c3.metric("Peak Consumption", f"{peak:,.3f}")
    c4.metric("Minimum Consumption", f"{minimum:,.3f}")

    st.markdown("<div class='energy-card'>", unsafe_allow_html=True)
    line_df = df[["datetime", target_col]].dropna().copy()
    fig_line = px.line(line_df, x="datetime", y=target_col, title="Electricity Usage Over Time")
    fig_line.update_traces(line_color="#0284c7")
    fig_line.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    daily = line_df.set_index("datetime")[target_col].resample("D").mean().reset_index()
    monthly = line_df.set_index("datetime")[target_col].resample("M").mean().reset_index()

    with t1:
        fig_daily = px.line(daily, x="datetime", y=target_col, title="Daily Consumption Trend")
        fig_daily.update_traces(line_color="#16a34a")
        st.plotly_chart(fig_daily, use_container_width=True)

    with t2:
        fig_monthly = px.bar(monthly, x="datetime", y=target_col, title="Monthly Average Consumption")
        fig_monthly.update_traces(marker_color="#14b8a6")
        st.plotly_chart(fig_monthly, use_container_width=True)

    by_hour = line_df.copy()
    by_hour["hour"] = by_hour["datetime"].dt.hour
    peak_hour = by_hour.groupby("hour")[target_col].mean().reset_index().sort_values(target_col, ascending=False).iloc[0]
    st.markdown(
        f"<div class='highlight'>Peak usage hour: {int(peak_hour['hour']):02d}:00 with average demand {peak_hour[target_col]:.3f}</div>",
        unsafe_allow_html=True,
    )


def render_data_analysis(df: pd.DataFrame, target_col: str, region_col: Optional[str]) -> None:
    st.title("📊 Data Analysis")
    st.caption("Slice the data with filters and inspect patterns, seasonality, and outliers.")

    min_date = df["datetime"].dt.date.min()
    max_date = df["datetime"].dt.date.max()

    f1, f2, f3, f4 = st.columns([2, 2, 2, 3])
    with f1:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with f2:
        hour_filter = st.slider("Hour filter", 0, 23, (0, 23))
    with f3:
        month_filter = st.multiselect("Months", options=list(range(1, 13)), default=list(range(1, 13)))
    with f4:
        if region_col and region_col in df.columns:
            regions = sorted(df[region_col].dropna().astype(str).unique().tolist())
            selected_regions = st.multiselect("Region/Zone", options=regions, default=regions)
        else:
            selected_regions = None
            st.markdown("<p class='small-note'>Region/zone column not available in this dataset.</p>", unsafe_allow_html=True)

    filtered = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[(filtered["datetime"] >= start_dt) & (filtered["datetime"] <= end_dt)]

    filtered = filtered[filtered["datetime"].dt.hour.between(hour_filter[0], hour_filter[1])]
    filtered = filtered[filtered["datetime"].dt.month.isin(month_filter)]

    if selected_regions is not None and region_col:
        filtered = filtered[filtered[region_col].astype(str).isin(selected_regions)]

    if filtered.empty:
        st.warning("No data in selected filter range.")
        return

    g1, g2 = st.columns(2)
    with g1:
        fig_ts = px.line(filtered, x="datetime", y=target_col, title="Time Series")
        fig_ts.update_traces(line_color="#0284c7")
        st.plotly_chart(fig_ts, use_container_width=True)
    with g2:
        fig_hist = px.histogram(filtered, x=target_col, nbins=60, title="Consumption Distribution")
        fig_hist.update_traces(marker_color="#16a34a")
        st.plotly_chart(fig_hist, use_container_width=True)

    h1, h2 = st.columns(2)
    with h1:
        fig_box = px.box(filtered, y=target_col, title="Outlier Detection (Box Plot)")
        fig_box.update_traces(marker_color="#14b8a6")
        st.plotly_chart(fig_box, use_container_width=True)

    with h2:
        numeric = filtered.select_dtypes(include=[np.number])
        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True)
            heat = px.imshow(
                corr,
                color_continuous_scale="GnBu",
                title="Correlation Heatmap",
                aspect="auto",
            )
            st.plotly_chart(heat, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

    by_hour = filtered.groupby(filtered["datetime"].dt.hour)[target_col].mean().sort_values(ascending=False)
    by_month = filtered.groupby(filtered["datetime"].dt.month)[target_col].mean().sort_values(ascending=False)

    q1, q3 = filtered[target_col].quantile([0.25, 0.75]).tolist()
    iqr = q3 - q1
    anomalies = filtered[(filtered[target_col] < (q1 - 1.5 * iqr)) | (filtered[target_col] > (q3 + 1.5 * iqr))]

    st.subheader("🔍 Insights")
    st.write(f"Peak usage hour: **{int(by_hour.index[0]):02d}:00**")
    st.write(f"Strongest month by average demand: **Month {int(by_month.index[0])}**")
    st.write(f"Potential anomalies detected: **{len(anomalies):,}** records")


def render_model_predictions(
    df: pd.DataFrame,
    target_col: str,
    feature_df: pd.DataFrame,
    models: Dict[str, object],
    feature_columns: List[str],
    scaler: Optional[object],
    temp_col: Optional[str],
) -> None:
    st.title("🤖 Model Predictions")
    st.caption("Provide input features and estimate future electricity demand with selected ML model.")

    if not models:
        st.error("No trained models found in models/Models folder.")
        return

    max_dt = df["datetime"].max()
    selected_date = st.date_input("Date", value=max_dt.date())
    selected_time = st.time_input("Time", value=max_dt.time().replace(second=0, microsecond=0))
    if selected_time is None:
        selected_time = max_dt.time().replace(second=0, microsecond=0)
    selected_dt = pd.Timestamp.combine(selected_date, selected_time)

    if temp_col and temp_col in df.columns and pd.api.types.is_numeric_dtype(df[temp_col]):
        t_default = float(df[temp_col].dropna().tail(1440).mean()) if not df[temp_col].dropna().empty else 20.0
    else:
        t_default = 20.0
    temperature_value = st.number_input("Temperature (if relevant)", value=float(t_default), format="%.2f")

    model_name = st.selectbox("Select model", options=list(models.keys()))

    with st.expander("Advanced feature overrides (optional)"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            m_lag_1 = st.number_input("lag_1", value=float(feature_df["lag_1"].iloc[-1]))
            m_lag_60 = st.number_input("lag_60", value=float(feature_df["lag_60"].iloc[-1]))
        with col_b:
            m_lag_2 = st.number_input("lag_2", value=float(feature_df["lag_2"].iloc[-1]))
            m_lag_1440 = st.number_input("lag_1440", value=float(feature_df["lag_1440"].iloc[-1]))
        with col_c:
            m_lag_3 = st.number_input("lag_3", value=float(feature_df["lag_3"].iloc[-1]))
            m_rm60 = st.number_input("rolling_mean_60", value=float(feature_df["rolling_mean_60"].iloc[-1]))

    if st.button("Predict", type="primary"):
        model = models[model_name]
        manual = {
            "lag_1": m_lag_1,
            "lag_2": m_lag_2,
            "lag_3": m_lag_3,
            "lag_60": m_lag_60,
            "lag_1440": m_lag_1440,
            "rolling_mean_60": m_rm60,
        }

        x = get_feature_row(selected_dt, feature_df, temperature_value, manual_lags=manual)
        x_aligned = align_features(x, feature_columns)

        pred_val = float(model_predict(model_name, model, x_aligned, scaler)[0])
        baseline = float(df[target_col].tail(60).mean())
        change_pct = ((pred_val - baseline) / baseline * 100) if baseline != 0 else 0.0

        st.markdown(
            f"<div class='highlight'>Predicted Electricity Demand ({model_name}): {pred_val:.4f}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Change vs last-hour average", f"{change_pct:+.2f}%")

        pred_df = pd.DataFrame(
            [
                {
                    "prediction_timestamp": selected_dt,
                    "model": model_name,
                    "predicted_demand": pred_val,
                }
            ]
        )
        csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download prediction as CSV",
            data=csv,
            file_name="electricity_prediction.csv",
            mime="text/csv",
        )


def render_model_comparison(
    feature_df: pd.DataFrame,
    models: Dict[str, object],
    feature_columns: List[str],
    scaler: Optional[object],
) -> None:
    st.title("📉 Model Comparison")
    st.caption("Compare MAE, RMSE, and SMAPE across available models.")

    if len(models) == 0:
        st.warning("No models available for comparison.")
        return

    metrics_df, pred_tables = evaluate_models(models, feature_df, feature_columns, scaler)
    if metrics_df.empty:
        st.warning("Unable to evaluate models with current artifacts.")
        return

    best_model = metrics_df.iloc[0]["Model"]
    st.markdown(
        f"<div class='highlight'>Best-performing model (lowest RMSE): {best_model}</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(metrics_df, use_container_width=True)

    for metric, color in [("MAE", "#0284c7"), ("RMSE", "#16a34a"), ("SMAPE", "#14b8a6")]:
        fig = px.bar(metrics_df, x="Model", y=metric, color="Model", title=f"{metric} Comparison")
        fig.update_traces(marker_color=color)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Visualization")
    horizon = st.slider("Forecast horizon (minutes)", min_value=30, max_value=7 * 24 * 60, value=7 * 24 * 60)
    chosen = st.selectbox("Model for forecast chart", list(models.keys()), index=list(models.keys()).index(best_model))

    forecast_df = recursive_forecast(chosen, models[chosen], feature_df, feature_columns, scaler, horizon)

    test_table = pred_tables.get(chosen)
    if test_table is not None and not test_table.empty:
        n = min(horizon, len(test_table))
        compare_actual = test_table.tail(n)[["datetime", "actual"]].copy()
        compare_pred = test_table.tail(n)[["datetime", "predicted"]].copy()

        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Scatter(
                x=compare_actual["datetime"],
                y=compare_actual["actual"],
                mode="lines",
                name="Actual",
                line=dict(color="#0284c7", width=2),
            )
        )
        fig_compare.add_trace(
            go.Scatter(
                x=compare_pred["datetime"],
                y=compare_pred["predicted"],
                mode="lines",
                name="Predicted",
                line=dict(color="#16a34a", width=2),
            )
        )
        fig_compare.update_layout(title="Predicted vs Actual (Backtest Window)", height=360)
        st.plotly_chart(fig_compare, use_container_width=True)

    fig_future = px.line(forecast_df, x="datetime", y="predicted", title="Future Forecast")
    fig_future.update_traces(line_color="#14b8a6")
    st.plotly_chart(fig_future, use_container_width=True)


def render_about() -> None:
    st.title("📌 About Project")
    st.markdown(
        """
        This Streamlit application provides an end-to-end workflow for electricity demand forecasting.

        ### Project Scope
        - Analyze and visualize electricity usage trends
        - Generate demand predictions from trained machine learning models
        - Compare model performance using MAE, RMSE, and SMAPE

        ### Tools & Technologies
        - Python
        - Streamlit
        - Pandas / NumPy
        - Scikit-learn
        - XGBoost
        - Plotly

        ### Role & Learning Outcomes
        - Built an interactive forecasting dashboard from data ingestion to model inference
        - Implemented robust preprocessing and model-agnostic feature alignment
        - Designed an analytics-first UI with actionable insights and visual storytelling
        """
    )


# -----------------------------
# App bootstrap
# -----------------------------
def main() -> None:
    st.sidebar.title("⚡ Energy Intelligence")
    dark_mode = st.sidebar.toggle("Dark mode", value=False)
    style_app(dark_mode)

    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard Overview",
            "Data Analysis",
            "Model Predictions",
            "Model Comparison",
            "About Project",
        ],
    )

    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    target_col = detect_target_column(df)
    optional_cols = infer_optional_columns(df)

    # Fill missing values for smoother analytics and modeling.
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].interpolate(limit_direction="both")

    feature_df = prepare_feature_frame(df, target_col)
    if feature_df.empty:
        st.error("Feature engineering produced an empty dataset. Check input data quality.")
        st.stop()

    models, feature_columns, scaler = load_models_and_meta()

    # Ensure Random Forest option exists even if pickle file is absent.
    if "Random Forest" not in models:
        rf = build_fallback_random_forest(feature_df, feature_columns)
        if rf is not None:
            models["Random Forest"] = rf

    if page == "Dashboard Overview":
        render_dashboard_overview(df, target_col)
    elif page == "Data Analysis":
        render_data_analysis(df, target_col, optional_cols.get("region"))
    elif page == "Model Predictions":
        render_model_predictions(df, target_col, feature_df, models, feature_columns, scaler, optional_cols.get("temperature"))
    elif page == "Model Comparison":
        render_model_comparison(feature_df, models, feature_columns, scaler)
    else:
        render_about()


if __name__ == "__main__":
    main()
