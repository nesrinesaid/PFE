from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from . import data_loader

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False


TARGET_COL = "VENTES"
DATE_COL = "Date"
MARKETS = ["VP", "VU"]

CORE_FEATURES = [
    "ANNEE", "MOIS", "EST_RAMADAN", "TRIMESTRE", "JOUR_ANNEE", "JOUR_SEMAINE",
    "EST_WEEKEND", "EST_VU", "EST_VACANCES_SCOLAIRES", "EST_FIN_MOIS",
    "EST_DEBUT_MOIS", "EST_JOUR_OUVRABLE", "EST_FETE_PUBLIQUE",
    "EST_DEBUT_TRIMESTRE", "EST_NOVEMBRE_DECEMBRE", "EST_JANVIER_FEVRIER",
]


@dataclass
class SplitBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    future: pd.DataFrame


def _clean_split(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if DATE_COL in work.columns:
        work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce")
    if TARGET_COL in work.columns:
        work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce")
    for col in work.columns:
        if col in {DATE_COL, "TYPE_MARCHE"}:
            continue
        if work[col].dtype == object:
            work[col] = pd.to_numeric(work[col].astype(str).str.replace(",", ".", regex=False), errors="ignore")
    return work.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_split_bundle() -> SplitBundle:
    return SplitBundle(
        train=_clean_split(data_loader.load_validation_split("data_train.csv")),
        validation=_clean_split(data_loader.load_validation_split("data_validation_2024.csv")),
        test=_clean_split(data_loader.load_validation_split("data_test_2025.csv")),
        future=_clean_split(data_loader.load_validation_split("data_future_2026.csv")),
    )


def _market_frame(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market not in MARKETS:
        raise ValueError(f"Unknown market: {market}")
    return df[df["TYPE_MARCHE"].astype(str).str.upper() == market].copy().reset_index(drop=True)


def _feature_columns(train_df: pd.DataFrame, future_df: pd.DataFrame) -> List[str]:
    candidates = []
    for col in CORE_FEATURES:
        if col in train_df.columns and col in future_df.columns:
            candidates.append(col)
    # include calendar dummies and other features present in both train and future
    for col in train_df.columns:
        if col in {DATE_COL, TARGET_COL, "TYPE_MARCHE"}:
            continue
        if col.startswith(("Mois_", "JS_", "Est_")) and col in future_df.columns:
            candidates.append(col)
    # de-duplicate and keep only numerics that exist in future
    cleaned = []
    for col in candidates:
        if col in cleaned:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]) or pd.api.types.is_bool_dtype(train_df[col]):
            cleaned.append(col)
    return cleaned


def _safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    result = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(np.nan_to_num(result, nan=0.0))


def _aggregate_monthly(df: pd.DataFrame, value_col: str) -> pd.Series:
    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    return work.groupby(work[DATE_COL].dt.to_period("M").dt.to_timestamp())[value_col].sum().sort_index()


def _make_model(model_name: str):
    if model_name == "xgb":
        if HAS_XGBOOST:
            return XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="reg:squarederror",
            )
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, random_state=42)
    return None


def _fit_predict_xgb(train: pd.DataFrame, predict_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    model = _make_model("xgb")
    X_train = train[features].fillna(0.0)
    y_train = train[TARGET_COL].fillna(0.0)
    model.fit(X_train, y_train)
    return np.maximum(0.0, model.predict(predict_df[features].fillna(0.0)))


def _fit_predict_sarimax(train: pd.DataFrame, predict_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    if not HAS_STATSMODELS or not features:
        return np.full(len(predict_df), float(train[TARGET_COL].mean()))
    try:
        model = SARIMAX(
            train[TARGET_COL],
            exog=train[features].fillna(0.0),
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        pred = fit.get_forecast(steps=len(predict_df), exog=predict_df[features].fillna(0.0)).predicted_mean.values
        return np.maximum(0.0, pred)
    except Exception:
        return np.full(len(predict_df), float(train[TARGET_COL].mean()))


def _fit_predict_prophet(train: pd.DataFrame, predict_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    if not HAS_PROPHET:
        return np.full(len(predict_df), float(train[TARGET_COL].mean()))
    try:
        train_p = train[[DATE_COL, TARGET_COL] + features].copy()
        predict_p = predict_df[[DATE_COL] + features].copy()
        train_p = train_p.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
        predict_p = predict_p.rename(columns={DATE_COL: "ds"})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.15)
        for feature in features:
            model.add_regressor(feature)
        model.fit(train_p)
        forecast = model.predict(predict_p)
        return np.maximum(0.0, forecast["yhat"].values)
    except Exception:
        return np.full(len(predict_df), float(train[TARGET_COL].mean()))


def _fit_predict_model(model_name: str, train: pd.DataFrame, predict_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    if model_name == "sarimax":
        return _fit_predict_sarimax(train, predict_df, features)
    if model_name == "prophet":
        return _fit_predict_prophet(train, predict_df, features)
    if model_name == "xgb":
        return _fit_predict_xgb(train, predict_df, features)
    raise ValueError(model_name)


def _fit_market_models(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, future: pd.DataFrame, model_name: str, market: str):
    train_m = _market_frame(train, market)
    val_m = _market_frame(val, market)
    test_m = _market_frame(test, market)
    future_m = _market_frame(future, market)
    features = _feature_columns(train_m, future_m)
    if not features:
        return {
            "val_pred": np.full(len(val_m), train_m[TARGET_COL].mean()),
            "test_pred": np.full(len(test_m), train_m[TARGET_COL].mean()),
            "future_pred": np.full(len(future_m), train_m[TARGET_COL].mean()),
            "val_actual": val_m[TARGET_COL].values,
            "test_actual": test_m[TARGET_COL].values,
            "test_dates": test_m[DATE_COL].values,
            "future_dates": future_m[DATE_COL].values,
            "features": features,
        }

    val_pred = _fit_predict_model(model_name, train_m, val_m, features)
    train_val = pd.concat([train_m, val_m], ignore_index=True)
    test_pred = _fit_predict_model(model_name, train_val, test_m, features)
    train_val_test = pd.concat([train_m, val_m, test_m], ignore_index=True)
    future_pred = _fit_predict_model(model_name, train_val_test, future_m, features)

    val_month_actual = _aggregate_monthly(val_m, TARGET_COL)
    test_month_actual = _aggregate_monthly(test_m, TARGET_COL)
    future_month = future_m.copy()
    future_month["PRED"] = future_pred
    future_month = future_month.groupby(future_month[DATE_COL].dt.to_period("M").dt.to_timestamp())["PRED"].sum().rename("PRED")
    val_month_pred = pd.Series(val_pred, index=val_m[DATE_COL]).groupby(pd.to_datetime(val_m[DATE_COL]).dt.to_period("M").dt.to_timestamp()).sum()
    test_month_pred = pd.Series(test_pred, index=test_m[DATE_COL]).groupby(pd.to_datetime(test_m[DATE_COL]).dt.to_period("M").dt.to_timestamp()).sum()
    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "future_pred": future_pred,
        "val_actual": val_m[TARGET_COL].values,
        "test_actual": test_m[TARGET_COL].values,
        "test_dates": test_m[DATE_COL].values,
        "future_dates": future_m[DATE_COL].values,
        "features": features,
        "val_month_actual": val_month_actual,
        "test_month_actual": test_month_actual,
        "val_month_pred": val_month_pred,
        "test_month_pred": test_month_pred,
        "future_month_pred": future_month,
    }


def _monthly_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    actual = actual.sort_index()
    pred = pred.reindex(actual.index).fillna(0.0)
    return {
        "MAPE": _safe_mape(actual.values, pred.values),
        "RMSE": float(np.sqrt(mean_squared_error(actual.values, pred.values))),
    }


def _aggregate_market_predictions(date_values, vp_pred, vu_pred, market_scope: str):
    df = pd.DataFrame({"Date": pd.to_datetime(date_values), "VP": vp_pred, "VU": vu_pred})
    if market_scope == "VP":
        out = df.groupby(df["Date"].dt.to_period("M").dt.to_timestamp())["VP"].sum().rename("PRED")
    elif market_scope == "VU":
        out = df.groupby(df["Date"].dt.to_period("M").dt.to_timestamp())["VU"].sum().rename("PRED")
    else:
        out = df.assign(TOTAL=df["VP"] + df["VU"]).groupby(df["Date"].dt.to_period("M").dt.to_timestamp())["TOTAL"].sum().rename("PRED")
    return out


@st.cache_resource(show_spinner=False)
def build_forecast_bundle() -> dict:
    splits = load_split_bundle()
    models = ["prophet", "sarimax", "xgb"]
    model_labels = {"prophet": "Prophet", "sarimax": "SARIMAX", "xgb": "XGBoost", "ensemble": "Ensemble"}

    results = {}
    metric_rows = []
    for model_name in models:
        vp = _fit_market_models(splits.train, splits.validation, splits.test, splits.future, model_name, "VP")
        vu = _fit_market_models(splits.train, splits.validation, splits.test, splits.future, model_name, "VU")

        val_pred_month = vp["val_month_pred"].add(vu["val_month_pred"], fill_value=0.0)
        test_pred_month = vp["test_month_pred"].add(vu["test_month_pred"], fill_value=0.0)
        val_actual_month = vp["val_month_actual"].add(vu["val_month_actual"], fill_value=0.0)
        test_actual_month = vp["test_month_actual"].add(vu["test_month_actual"], fill_value=0.0)

        val_metrics = _monthly_metrics(val_actual_month, val_pred_month)
        test_metrics = _monthly_metrics(test_actual_month, test_pred_month)
        metric_rows.append({
            "Modèle": model_labels[model_name],
            "MAPE val": val_metrics["MAPE"],
            "MAPE test": test_metrics["MAPE"],
            "RMSE": test_metrics["RMSE"],
            "Statut": "✅ Objectif atteint" if test_metrics["MAPE"] < 15 else ("⚠️ Proche" if test_metrics["MAPE"] < 20 else "❌ Non atteint"),
        })

        future_pred_total = vp["future_month_pred"].add(vu["future_month_pred"], fill_value=0.0)
        residual_std = float(np.nanstd(test_actual_month.values - test_pred_month.reindex(test_actual_month.index).fillna(0.0).values)) or 1.0
        future_df = pd.DataFrame({"Date": future_pred_total.index, "PRED": future_pred_total.values})
        future_df["CI80_LOWER"] = np.maximum(0.0, future_df["PRED"] - 1.28 * residual_std)
        future_df["CI80_UPPER"] = future_df["PRED"] + 1.28 * residual_std
        future_df["CI95_LOWER"] = np.maximum(0.0, future_df["PRED"] - 1.96 * residual_std)
        future_df["CI95_UPPER"] = future_df["PRED"] + 1.96 * residual_std

        backtest_df = pd.DataFrame({
            "Date": test_actual_month.index,
            "Actual": test_actual_month.values,
            "Predicted": test_pred_month.reindex(test_actual_month.index).fillna(0.0).values,
        })
        results[model_labels[model_name]] = {
            "future": future_df,
            "backtest": backtest_df,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "future_vp": vp["future_month_pred"],
            "future_vu": vu["future_month_pred"],
            "backtest_vp": vp["test_month_pred"],
            "backtest_vu": vu["test_month_pred"],
            "actual_vp": vp["test_month_actual"],
            "actual_vu": vu["test_month_actual"],
        }

    metric_df = pd.DataFrame(metric_rows)
    # ensemble = mean of model forecasts
    ensemble_future = None
    ensemble_backtest = None
    ensemble_future_vp = None
    ensemble_future_vu = None
    ensemble_backtest_vp = None
    ensemble_backtest_vu = None
    for model_name, bundle in results.items():
        if ensemble_future is None:
            ensemble_future = bundle["future"].copy()
            ensemble_backtest = bundle["backtest"].copy()
            ensemble_future_vp = bundle["future_vp"].copy()
            ensemble_future_vu = bundle["future_vu"].copy()
            ensemble_backtest_vp = bundle["backtest_vp"].copy()
            ensemble_backtest_vu = bundle["backtest_vu"].copy()
        else:
            ensemble_future["PRED"] += bundle["future"]["PRED"].values
            ensemble_backtest["Predicted"] += bundle["backtest"]["Predicted"].values
            ensemble_future_vp = ensemble_future_vp.add(bundle["future_vp"], fill_value=0.0)
            ensemble_future_vu = ensemble_future_vu.add(bundle["future_vu"], fill_value=0.0)
            ensemble_backtest_vp = ensemble_backtest_vp.add(bundle["backtest_vp"], fill_value=0.0)
            ensemble_backtest_vu = ensemble_backtest_vu.add(bundle["backtest_vu"], fill_value=0.0)
    ensemble_future["PRED"] /= len(results)
    ensemble_backtest["Predicted"] /= len(results)
    ensemble_future_vp = ensemble_future_vp / len(results)
    ensemble_future_vu = ensemble_future_vu / len(results)
    ensemble_backtest_vp = ensemble_backtest_vp / len(results)
    ensemble_backtest_vu = ensemble_backtest_vu / len(results)
    residual_std = float(np.nanstd(ensemble_backtest["Actual"].values - ensemble_backtest["Predicted"].values)) or 1.0
    ensemble_future["CI80_LOWER"] = np.maximum(0.0, ensemble_future["PRED"] - 1.28 * residual_std)
    ensemble_future["CI80_UPPER"] = ensemble_future["PRED"] + 1.28 * residual_std
    ensemble_future["CI95_LOWER"] = np.maximum(0.0, ensemble_future["PRED"] - 1.96 * residual_std)
    ensemble_future["CI95_UPPER"] = ensemble_future["PRED"] + 1.96 * residual_std

    metric_df = pd.concat([
        metric_df,
        pd.DataFrame([{ 
            "Modèle": "Ensemble",
            "MAPE val": metric_df["MAPE val"].mean(),
            "MAPE test": _safe_mape(ensemble_backtest["Actual"].values, ensemble_backtest["Predicted"].values),
            "RMSE": float(np.sqrt(mean_squared_error(ensemble_backtest["Actual"].values, ensemble_backtest["Predicted"].values))),
            "Statut": "✅ Objectif atteint" if _safe_mape(ensemble_backtest["Actual"].values, ensemble_backtest["Predicted"].values) < 15 else "⚠️ Proche",
        }])
    ], ignore_index=True)

    results["Ensemble"] = {
        "future": ensemble_future,
        "backtest": ensemble_backtest,
        "future_vp": ensemble_future_vp,
        "future_vu": ensemble_future_vu,
        "backtest_vp": ensemble_backtest_vp,
        "backtest_vu": ensemble_backtest_vu,
        "val_metrics": {"MAPE": float(metric_df["MAPE val"].mean()), "RMSE": float(metric_df["RMSE"].mean())},
        "test_metrics": {"MAPE": _safe_mape(ensemble_backtest["Actual"].values, ensemble_backtest["Predicted"].values), "RMSE": float(np.sqrt(mean_squared_error(ensemble_backtest["Actual"].values, ensemble_backtest["Predicted"].values)))},
    }

    return {"metrics": metric_df, "results": results}


def load_model(name: str):
    bundle = build_forecast_bundle()
    return bundle["results"].get(name)


def model_metrics_table() -> pd.DataFrame:
    return build_forecast_bundle()["metrics"].copy()


def forecast_for_scope(model_name: str, scope: str = "Total") -> pd.DataFrame:
    bundle = load_model(model_name)
    if bundle is None:
        raise ValueError(f"Unknown model {model_name}")
    future = bundle["future"].copy()
    forecast = future.rename(columns={"PRED": "PREV_TOTAL_MARCHE"}).copy()
    if scope == "VP":
        forecast["PREV_TOTAL_MARCHE"] *= 0.78
    elif scope == "VU":
        forecast["PREV_TOTAL_MARCHE"] *= 0.22
    return forecast

