import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

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

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


RAMADAN_RANGES = [
    ("2019-05-06", "2019-06-04"),
    ("2020-04-24", "2020-05-23"),
    ("2021-04-13", "2021-05-12"),
    ("2022-04-02", "2022-05-01"),
    ("2023-03-23", "2023-04-21"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-30"),
    ("2026-02-18", "2026-03-19"),
]

ARTES_BRANDS = {"RENAULT", "DACIA", "NISSAN"}


@dataclass
class ModelResult:
    name: str
    y_pred: pd.Series
    mae: float
    rmse: float
    smape: float


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return 100.0 * np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask])


def month_ramadan_days(month_start):
    month_start = pd.Timestamp(month_start).normalize().replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)
    total_days = 0
    for start, end in RAMADAN_RANGES:
        r_start = pd.Timestamp(start)
        r_end = pd.Timestamp(end)
        overlap_start = max(month_start, r_start)
        overlap_end = min(month_end, r_end)
        if overlap_end >= overlap_start:
            total_days += (overlap_end - overlap_start).days + 1
    return int(total_days)


def ensure_monthly_index(df, date_col="Date"):
    if df.empty:
        return df
    start = df[date_col].min().replace(day=1)
    end = df[date_col].max().replace(day=1)
    full_months = pd.date_range(start, end, freq="MS")
    return df.set_index(date_col).reindex(full_months).rename_axis(date_col).reset_index()


def load_monthly_base(project_root):
    path = os.path.join(project_root, "data_cleaned_enriched.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("data_cleaned_enriched.csv not found. Run step2_5_enrich_data.py first.")

    df = pd.read_csv(path, parse_dates=["DATV"])
    if "IM_RI" not in df.columns:
        raise ValueError("IM_RI column missing in data_cleaned_enriched.csv")

    if "TYPE_MARCHE" not in df.columns:
        if "Marché" in df.columns:
            df["TYPE_MARCHE"] = df["Marché"].astype(str).str.upper().str.extract(r"(VP|VU)", expand=False)
        else:
            raise ValueError("TYPE_MARCHE column missing and could not be derived")

    df["IM_RI"] = pd.to_numeric(df["IM_RI"], errors="coerce").round().astype("Int64")
    df = df[df["IM_RI"].eq(10)].copy()
    df = df[df["TYPE_MARCHE"].isin(["VP", "VU"])].copy()
    df["Date"] = pd.to_datetime(df["DATV"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["Date"]).copy()

    by_market = (
        df.groupby(["Date", "TYPE_MARCHE"]).size().reset_index(name="VENTES")
        .pivot(index="Date", columns="TYPE_MARCHE", values="VENTES")
        .fillna(0.0)
        .reset_index()
    )
    for col in ["VP", "VU"]:
        if col not in by_market.columns:
            by_market[col] = 0.0
    by_market = ensure_monthly_index(by_market, "Date")
    by_market[["VP", "VU"]] = by_market[["VP", "VU"]].fillna(0.0)
    by_market["TOTAL"] = by_market["VP"] + by_market["VU"]
    by_market["PART_VP"] = np.where(by_market["TOTAL"] > 0, by_market["VP"] / by_market["TOTAL"], 0.5)
    by_market["PART_VU"] = 1.0 - by_market["PART_VP"]
    by_market["RAMADAN_DAYS"] = by_market["Date"].map(month_ramadan_days)
    by_market["EST_RAMADAN"] = (by_market["RAMADAN_DAYS"] > 0).astype(int)
    by_market["MOIS"] = by_market["Date"].dt.month
    by_market["ANNEE"] = by_market["Date"].dt.year

    # ARTES brands monthly volume + share over total.
    if "MARQUE" in df.columns:
        df["MARQUE_UP"] = df["MARQUE"].astype(str).str.upper().str.strip()
        arts = (
            df[df["MARQUE_UP"].isin(ARTES_BRANDS)]
            .groupby("Date")
            .size()
            .rename("ARTES_VOL")
            .reset_index()
        )
    else:
        arts = pd.DataFrame({"Date": by_market["Date"], "ARTES_VOL": 0.0})

    by_market = by_market.merge(arts, on="Date", how="left")
    by_market["ARTES_VOL"] = by_market["ARTES_VOL"].fillna(0.0)
    by_market["ARTES_SHARE"] = np.where(by_market["TOTAL"] > 0, by_market["ARTES_VOL"] / by_market["TOTAL"], 0.0)

    return df, by_market


def load_external_features(project_root):
    path = os.path.join(project_root, "data", "donnees_externes_tunisie.xlsx")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Date"])

    try:
        xls = pd.ExcelFile(path)
        if not xls.sheet_names:
            return pd.DataFrame(columns=["Date"])
        ext = pd.read_excel(path, sheet_name=xls.sheet_names[0])
    except Exception:
        return pd.DataFrame(columns=["Date"])

    ext.columns = [str(c).strip().upper() for c in ext.columns]

    date_col = None
    for c in ext.columns:
        if "DATE" in c or "MOIS" in c:
            date_col = c
            break

    if date_col is None and {"ANNEE", "MOIS"}.issubset(set(ext.columns)):
        ext["Date"] = pd.to_datetime(
            ext["ANNEE"].astype(str) + "-" + ext["MOIS"].astype(str) + "-01", errors="coerce"
        )
    elif date_col is not None:
        ext["Date"] = pd.to_datetime(ext[date_col], errors="coerce")
    else:
        # Fallback: infer yearly rows then align to January.
        if "ANNEE" in ext.columns:
            ext["Date"] = pd.to_datetime(ext["ANNEE"].astype(str) + "-01-01", errors="coerce")
        else:
            return pd.DataFrame(columns=["Date"])

    ext["Date"] = ext["Date"].dt.to_period("M").dt.to_timestamp()
    ext = ext.dropna(subset=["Date"]).copy()

    numeric_cols = []
    for c in ext.columns:
        if c == "Date":
            continue
        series = pd.to_numeric(ext[c], errors="coerce")
        if series.notna().sum() > 0:
            ext[c] = series
            numeric_cols.append(c)

    if not numeric_cols:
        return pd.DataFrame(columns=["Date"])

    keep_cols = ["Date"] + numeric_cols[:6]
    ext = ext[keep_cols].groupby("Date", as_index=False).mean()
    return ext


def load_scenarios(project_root):
    """Load scenario sheets (baseline/optimiste/prudent or sheets starting with 'scenario')
    Returns dict: {sheet_name: df}
    """
    path = os.path.join(project_root, "data", "donnees_externes_tunisie.xlsx")
    if not os.path.exists(path):
        return {}
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return {}

    scenarios = {}
    for name in xls.sheet_names:
        nup = name.strip().lower()
        if any(k in nup for k in ("scenario", "baseline", "optimiste", "pessimiste", "prudent", "conserv")):
            try:
                df = pd.read_excel(path, sheet_name=name)
                df.columns = [str(c).strip().upper() for c in df.columns]
                # Try to convert to Date + numeric columns (reuse load_external_features logic)
                date_col = None
                for c in df.columns:
                    if "DATE" in c or "MOIS" in c:
                        date_col = c
                        break
                if date_col is None and {"ANNEE", "MOIS"}.issubset(set(df.columns)):
                    df["Date"] = pd.to_datetime(df["ANNEE"].astype(str) + "-" + df["MOIS"].astype(str) + "-01", errors="coerce")
                elif date_col is not None:
                    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
                else:
                    if "ANNEE" in df.columns:
                        df["Date"] = pd.to_datetime(df["ANNEE"].astype(str) + "-01-01", errors="coerce")
                    else:
                        continue

                df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
                df = df.dropna(subset=["Date"]).copy()
                numeric_cols = []
                for c in df.columns:
                    if c == "Date":
                        continue
                    series = pd.to_numeric(df[c], errors="coerce")
                    if series.notna().sum() > 0:
                        df[c] = series
                        numeric_cols.append(c)
                if not numeric_cols:
                    continue
                keep_cols = ["Date"] + numeric_cols[:6]
                scenarios[name] = df[keep_cols].groupby("Date", as_index=False).mean()
            except Exception:
                continue
    return scenarios


def add_lags_and_ma(df, col, lags=(1, 3, 6, 12), mas=(3, 6, 12)):
    out = df.copy()
    for lag in lags:
        out[f"{col}_LAG{lag}"] = out[col].shift(lag)
    for w in mas:
        out[f"{col}_MA{w}"] = out[col].shift(1).rolling(w, min_periods=1).mean()
    return out


def make_share_supervised(df, share_col, extra_cols=None):
    if extra_cols is None:
        extra_cols = []
    tmp = add_lags_and_ma(df, share_col)
    tmp = add_lags_and_ma(tmp, "TOTAL", lags=(1, 3, 6, 12), mas=(3, 6))
    tmp["MONTH_SIN"] = np.sin(2 * np.pi * tmp["MOIS"] / 12.0)
    tmp["MONTH_COS"] = np.cos(2 * np.pi * tmp["MOIS"] / 12.0)

    features = [
        f"{share_col}_LAG1", f"{share_col}_LAG3", f"{share_col}_LAG6", f"{share_col}_LAG12",
        f"{share_col}_MA3", f"{share_col}_MA6", f"{share_col}_MA12",
        "TOTAL_LAG1", "TOTAL_LAG3", "TOTAL_LAG6", "TOTAL_LAG12", "TOTAL_MA3", "TOTAL_MA6",
        "RAMADAN_DAYS", "EST_RAMADAN", "MONTH_SIN", "MONTH_COS",
    ]
    features += [c for c in extra_cols if c in tmp.columns]
    features = list(dict.fromkeys(features))

    target = share_col
    tmp = tmp.dropna(subset=features + [target]).copy()
    return tmp, features, target


def build_ml_regressor():
    if HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=500, random_state=42)


def fit_predict_sarimax(train_df, test_df, y_col, exog_cols):
    if not HAS_STATSMODELS:
        return None
    try:
        exog_train = train_df[exog_cols] if exog_cols else None
        exog_test = test_df[exog_cols] if exog_cols else None
        model = SARIMAX(
            train_df[y_col],
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        pred = fit.get_forecast(steps=len(test_df), exog=exog_test).predicted_mean
        return pd.Series(np.maximum(0.0, pred.values), index=test_df["Date"])
    except Exception:
        return None


def fit_predict_prophet(train_df, test_df, y_col, reg_cols):
    if not HAS_PROPHET:
        return None
    try:
        p_train = train_df[["Date", y_col] + reg_cols].rename(columns={"Date": "ds", y_col: "y"}).copy()
        p_test = test_df[["Date"] + reg_cols].rename(columns={"Date": "ds"}).copy()
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.2,
        )
        for c in reg_cols:
            m.add_regressor(c)
        m.fit(p_train)
        pred = m.predict(p_test)["yhat"].values
        return pd.Series(np.maximum(0.0, pred), index=test_df["Date"])
    except Exception:
        return None


def evaluate_model(name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return ModelResult(
        name=name,
        y_pred=pd.Series(y_pred),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        smape=float(smape(y_true, y_pred)),
    )


def one_step_share_features(hist, date, share_col, extra_cols):
    idx = len(hist) - 1
    month = pd.Timestamp(date).month
    feat = {
        f"{share_col}_LAG1": hist[share_col].iloc[idx],
        f"{share_col}_LAG3": hist[share_col].iloc[max(0, idx - 2): idx + 1].mean(),
        f"{share_col}_LAG6": hist[share_col].iloc[max(0, idx - 5): idx + 1].mean(),
        f"{share_col}_LAG12": hist[share_col].iloc[max(0, idx - 11): idx + 1].mean(),
        f"{share_col}_MA3": hist[share_col].iloc[max(0, idx - 2): idx + 1].mean(),
        f"{share_col}_MA6": hist[share_col].iloc[max(0, idx - 5): idx + 1].mean(),
        f"{share_col}_MA12": hist[share_col].iloc[max(0, idx - 11): idx + 1].mean(),
        "TOTAL_LAG1": hist["TOTAL"].iloc[idx],
        "TOTAL_LAG3": hist["TOTAL"].iloc[max(0, idx - 2): idx + 1].mean(),
        "TOTAL_LAG6": hist["TOTAL"].iloc[max(0, idx - 5): idx + 1].mean(),
        "TOTAL_LAG12": hist["TOTAL"].iloc[max(0, idx - 11): idx + 1].mean(),
        "TOTAL_MA3": hist["TOTAL"].iloc[max(0, idx - 2): idx + 1].mean(),
        "TOTAL_MA6": hist["TOTAL"].iloc[max(0, idx - 5): idx + 1].mean(),
        "RAMADAN_DAYS": month_ramadan_days(date),
        "EST_RAMADAN": int(month_ramadan_days(date) > 0),
        "MONTH_SIN": np.sin(2 * np.pi * month / 12.0),
        "MONTH_COS": np.cos(2 * np.pi * month / 12.0),
    }
    for c in extra_cols:
        feat[c] = hist[c].iloc[idx] if c in hist.columns else 0.0
    return feat


def build_city_watch(df_raw):
    tmp = df_raw.copy()
    tmp["ANNEE"] = pd.to_datetime(tmp["DATV"]).dt.year
    tmp["MOIS"] = pd.to_datetime(tmp["DATV"]).dt.month
    tmp = tmp[tmp["MOIS"].between(1, 6)]

    v2024 = (
        tmp[tmp["ANNEE"] == 2024]
        .groupby("VILLE")
        .size()
        .rename("VENTES_S1_2024")
        .reset_index()
    )
    v2025 = (
        tmp[tmp["ANNEE"] == 2025]
        .groupby("VILLE")
        .size()
        .rename("VENTES_S1_2025")
        .reset_index()
    )

    watch = v2025.merge(v2024, on="VILLE", how="outer").fillna(0.0)
    watch["CROISSANCE_%"] = np.where(
        watch["VENTES_S1_2024"] > 0,
        100.0 * (watch["VENTES_S1_2025"] - watch["VENTES_S1_2024"]) / watch["VENTES_S1_2024"],
        np.nan,
    )
    watch = watch.sort_values(["VENTES_S1_2025", "CROISSANCE_%"], ascending=[False, False]).reset_index(drop=True)
    return watch


def main():
    print("STEP 6 - MODELING & FORECASTING (S1 2026)\n")
    project_root = os.path.dirname(os.path.abspath(__file__))

    df_raw, df_month = load_monthly_base(project_root)
    # load single-sheet external features (legacy) and also scenario sheets (preferred)
    ext_default = load_external_features(project_root)
    scenarios = load_scenarios(project_root)

    # If there are scenario sheets, we'll run the forecasting per scenario.
    if not scenarios:
        # No scenario sheets: fall back to single external sheet behaviour
        ext = ext_default
        if not ext.empty:
            ext_non_date = [c for c in ext.columns if c != "Date"]
            ext_non_conflict = [c for c in ext_non_date if c not in df_month.columns]
            ext_use = ext[["Date"] + ext_non_conflict].copy()
            df_month = df_month.merge(ext_use, on="Date", how="left")
            for c in ext.columns:
                if c == "Date":
                    continue
                if c in df_month.columns:
                    df_month[c] = df_month[c].ffill().bfill()

    def run_forecasting(df_month_local, df_raw_local, scenario_label=None):
        """Core forecasting logic turned into a function; writes outputs with optional scenario suffix."""
        label = f"_{scenario_label}" if scenario_label else ""

        # Train/test for model comparison: test = 2025.
        train = df_month_local[df_month_local["Date"] < pd.Timestamp("2025-01-01")].copy()
        test = df_month_local[(df_month_local["Date"] >= pd.Timestamp("2025-01-01")) & (df_month_local["Date"] <= pd.Timestamp("2025-12-01"))].copy()
        if train.empty or test.empty:
            raise ValueError("Insufficient monthly data for train/test split (needs 2019-2025)")

        exog_cols = [c for c in ["RAMADAN_DAYS", "EST_RAMADAN"] if c in df_month_local.columns]
        for c in df_month_local.columns:
            if c in {"Date", "VP", "VU", "TOTAL", "PART_VP", "PART_VU", "ARTES_VOL", "ARTES_SHARE", "MOIS", "ANNEE"}:
                continue
            if pd.api.types.is_numeric_dtype(df_month_local[c]) and c not in exog_cols:
                exog_cols.append(c)
        exog_cols = exog_cols[:6]

        # Model 1: monthly total market.
        print("1) Forecast total monthly market (VP+VU)...")
        sarimax_pred = fit_predict_sarimax(train, test, "TOTAL", exog_cols)
        prophet_pred = fit_predict_prophet(train, test, "TOTAL", exog_cols)

        model_results = []
        if sarimax_pred is not None:
            model_results.append(evaluate_model("SARIMAX_TOTAL", test["TOTAL"], sarimax_pred.values))
        if prophet_pred is not None:
            model_results.append(evaluate_model("PROPHET_TOTAL", test["TOTAL"], prophet_pred.values))

        if sarimax_pred is not None and prophet_pred is not None:
            ensemble_test = 0.6 * sarimax_pred.values + 0.4 * prophet_pred.values
        elif sarimax_pred is not None:
            ensemble_test = sarimax_pred.values
        elif prophet_pred is not None:
            ensemble_test = prophet_pred.values
        else:
            seasonal = train.set_index("Date")["TOTAL"]
            ensemble_test = []
            for d in test["Date"]:
                prev = d - pd.DateOffset(years=1)
                ensemble_test.append(float(seasonal.loc[prev]) if prev in seasonal.index else float(train["TOTAL"].mean()))
            ensemble_test = np.array(ensemble_test)

        model_results.append(evaluate_model("ENSEMBLE_TOTAL", test["TOTAL"], ensemble_test))

        # Build future total forecast for S1 2026.
        future_dates = pd.date_range("2026-01-01", "2026-06-01", freq="MS")
        df_future = pd.DataFrame({"Date": future_dates})
        df_future["RAMADAN_DAYS"] = df_future["Date"].map(month_ramadan_days)
        df_future["EST_RAMADAN"] = (df_future["RAMADAN_DAYS"] > 0).astype(int)
        for c in exog_cols:
            if c in df_future.columns:
                continue
            if c in df_month_local.columns:
                df_future[c] = df_month_local[c].iloc[-1]
            else:
                df_future[c] = 0.0

        total_future_sarimax = None
        total_future_prophet = None

        if HAS_STATSMODELS:
            try:
                model = SARIMAX(
                    df_month_local["TOTAL"],
                    exog=df_month_local[exog_cols] if exog_cols else None,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False)
                total_future_sarimax = fit.get_forecast(steps=len(df_future), exog=df_future[exog_cols]).predicted_mean.values
            except Exception:
                total_future_sarimax = None

        if HAS_PROPHET:
            try:
                p_train = df_month_local[["Date", "TOTAL"] + exog_cols].rename(columns={"Date": "ds", "TOTAL": "y"})
                p_future = df_future[["Date"] + exog_cols].rename(columns={"Date": "ds"})
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.2)
                for c in exog_cols:
                    m.add_regressor(c)
                m.fit(p_train)
                total_future_prophet = m.predict(p_future)["yhat"].values
            except Exception:
                total_future_prophet = None

        if total_future_sarimax is not None and total_future_prophet is not None:
            df_future["TOTAL_PRED"] = np.maximum(0.0, 0.6 * total_future_sarimax + 0.4 * total_future_prophet)
        elif total_future_sarimax is not None:
            df_future["TOTAL_PRED"] = np.maximum(0.0, total_future_sarimax)
        elif total_future_prophet is not None:
            df_future["TOTAL_PRED"] = np.maximum(0.0, total_future_prophet)
        else:
            season_map = df_month_local.set_index(df_month_local["Date"].dt.month)["TOTAL"].groupby(level=0).mean()
            df_future["TOTAL_PRED"] = df_future["Date"].dt.month.map(season_map).fillna(df_month_local["TOTAL"].mean()).values

        return {
            "train": train,
            "test": test,
            "exog_cols": exog_cols,
            "df_future": df_future,
            "df_month": df_month_local,
            "df_raw": df_raw_local,
            "model_results": model_results,
            "sarimax_pred": sarimax_pred,
            "prophet_pred": prophet_pred,
            "ensemble_test": ensemble_test,
            "label": label,
        }

    # If we have scenarios, run forecasts per scenario and save combined outputs.
    if scenarios:
        print(f"Found scenario sheets: {list(scenarios.keys())}")
        for name, ext_df in scenarios.items():
            dfm = df_month.copy()
            ext_non_date = [c for c in ext_df.columns if c != "Date"]
            ext_non_conflict = [c for c in ext_non_date if c not in dfm.columns]
            if ext_non_conflict:
                dfm = dfm.merge(ext_df[["Date"] + ext_non_conflict], on="Date", how="left")
                for c in ext_df.columns:
                    if c == "Date":
                        continue
                    if c in dfm.columns:
                        dfm[c] = dfm[c].ffill().bfill()

            res = run_forecasting(dfm, df_raw, scenario_label=name)
            out_forecast = os.path.join(project_root, f"step6_forecast_s1_2026_{name}.csv")
            res["df_future"].rename(columns={"TOTAL_PRED": "PREV_TOTAL_MARCHE"})[["Date", "PREV_TOTAL_MARCHE"]].to_csv(out_forecast, index=False)
            print(f"Saved scenario forecast: {os.path.basename(out_forecast)}")
            out_metrics = os.path.join(project_root, f"step6_metrics_summary_{name}.csv")
            pd.DataFrame([{"name": m.name, "mae": m.mae, "rmse": m.rmse, "smape": m.smape} for m in res["model_results"]]).to_csv(out_metrics, index=False)
            print(f"Saved scenario metrics: {os.path.basename(out_metrics)}")

        print("Scenario runs complete. Individual CSVs saved for each scenario.")
        return

    # No scenarios: continue old single-run behaviour (df_month already possibly merged above)

    # Train/test for model comparison: test = 2025.
    train = df_month[df_month["Date"] < pd.Timestamp("2025-01-01")].copy()
    test = df_month[(df_month["Date"] >= pd.Timestamp("2025-01-01")) & (df_month["Date"] <= pd.Timestamp("2025-12-01"))].copy()
    if train.empty or test.empty:
        raise ValueError("Insufficient monthly data for train/test split (needs 2019-2025)")

    exog_cols = [c for c in ["RAMADAN_DAYS", "EST_RAMADAN"] if c in df_month.columns]
    for c in df_month.columns:
        if c in {"Date", "VP", "VU", "TOTAL", "PART_VP", "PART_VU", "ARTES_VOL", "ARTES_SHARE", "MOIS", "ANNEE"}:
            continue
        if pd.api.types.is_numeric_dtype(df_month[c]) and c not in exog_cols:
            exog_cols.append(c)
    exog_cols = exog_cols[:6]

    # Model 1: monthly total market.
    print("1) Forecast total monthly market (VP+VU)...")
    sarimax_pred = fit_predict_sarimax(train, test, "TOTAL", exog_cols)
    prophet_pred = fit_predict_prophet(train, test, "TOTAL", exog_cols)

    model_results = []
    if sarimax_pred is not None:
        model_results.append(evaluate_model("SARIMAX_TOTAL", test["TOTAL"], sarimax_pred.values))
    if prophet_pred is not None:
        model_results.append(evaluate_model("PROPHET_TOTAL", test["TOTAL"], prophet_pred.values))

    if sarimax_pred is not None and prophet_pred is not None:
        ensemble_test = 0.6 * sarimax_pred.values + 0.4 * prophet_pred.values
    elif sarimax_pred is not None:
        ensemble_test = sarimax_pred.values
    elif prophet_pred is not None:
        ensemble_test = prophet_pred.values
    else:
        # Fallback: naive seasonal baseline.
        seasonal = train.set_index("Date")["TOTAL"]
        ensemble_test = []
        for d in test["Date"]:
            prev = d - pd.DateOffset(years=1)
            ensemble_test.append(float(seasonal.loc[prev]) if prev in seasonal.index else float(train["TOTAL"].mean()))
        ensemble_test = np.array(ensemble_test)

    model_results.append(evaluate_model("ENSEMBLE_TOTAL", test["TOTAL"], ensemble_test))

    # Build future total forecast for S1 2026.
    future_dates = pd.date_range("2026-01-01", "2026-06-01", freq="MS")
    df_future = pd.DataFrame({"Date": future_dates})
    df_future["RAMADAN_DAYS"] = df_future["Date"].map(month_ramadan_days)
    df_future["EST_RAMADAN"] = (df_future["RAMADAN_DAYS"] > 0).astype(int)
    for c in exog_cols:
        if c in df_future.columns:
            continue
        if c in df_month.columns:
            df_future[c] = df_month[c].iloc[-1]
        else:
            df_future[c] = 0.0

    total_future_sarimax = None
    total_future_prophet = None

    if HAS_STATSMODELS:
        try:
            model = SARIMAX(
                df_month["TOTAL"],
                exog=df_month[exog_cols] if exog_cols else None,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            total_future_sarimax = fit.get_forecast(steps=len(df_future), exog=df_future[exog_cols]).predicted_mean.values
        except Exception:
            total_future_sarimax = None

    if HAS_PROPHET:
        try:
            p_train = df_month[["Date", "TOTAL"] + exog_cols].rename(columns={"Date": "ds", "TOTAL": "y"})
            p_future = df_future[["Date"] + exog_cols].rename(columns={"Date": "ds"})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.2)
            for c in exog_cols:
                m.add_regressor(c)
            m.fit(p_train)
            total_future_prophet = m.predict(p_future)["yhat"].values
        except Exception:
            total_future_prophet = None

    if total_future_sarimax is not None and total_future_prophet is not None:
        df_future["TOTAL_PRED"] = np.maximum(0.0, 0.6 * total_future_sarimax + 0.4 * total_future_prophet)
    elif total_future_sarimax is not None:
        df_future["TOTAL_PRED"] = np.maximum(0.0, total_future_sarimax)
    elif total_future_prophet is not None:
        df_future["TOTAL_PRED"] = np.maximum(0.0, total_future_prophet)
    else:
        # Seasonal fallback.
        season_map = df_month.set_index(df_month["Date"].dt.month)["TOTAL"].groupby(level=0).mean()
        df_future["TOTAL_PRED"] = df_future["Date"].dt.month.map(season_map).fillna(df_month["TOTAL"].mean()).values

    # Model 2: VP/VU decomposition via share model.
    print("2) Forecast VP/VU decomposition...")
    extra_share_cols = [c for c in exog_cols if c not in {"RAMADAN_DAYS", "EST_RAMADAN"}]
    sup_share, share_features, share_target = make_share_supervised(df_month, "PART_VP", extra_share_cols)
    share_train = sup_share[sup_share["Date"] < pd.Timestamp("2025-01-01")].copy()
    share_test = sup_share[(sup_share["Date"] >= pd.Timestamp("2025-01-01")) & (sup_share["Date"] <= pd.Timestamp("2025-12-01"))].copy()

    share_model = build_ml_regressor()
    if not share_train.empty:
        share_model.fit(share_train[share_features], share_train[share_target])

    if not share_test.empty:
        share_test_pred = np.clip(share_model.predict(share_test[share_features]), 0.05, 0.95)
        vp_test_pred = share_test_pred * share_test["TOTAL"].values
        vu_test_pred = share_test["TOTAL"].values - vp_test_pred
        model_results.append(evaluate_model("XGB_SHARE_VP", share_test["PART_VP"], share_test_pred))
        model_results.append(evaluate_model("VP_FROM_SHARE", test["VP"], vp_test_pred))
        model_results.append(evaluate_model("VU_FROM_SHARE", test["VU"], vu_test_pred))

    # Recursive share forecast for S1 2026.
    hist = df_month.copy().reset_index(drop=True)
    for d, total_pred in zip(df_future["Date"], df_future["TOTAL_PRED"]):
        next_row = {
            "Date": d,
            "TOTAL": float(total_pred),
            "RAMADAN_DAYS": month_ramadan_days(d),
            "EST_RAMADAN": int(month_ramadan_days(d) > 0),
            "MOIS": pd.Timestamp(d).month,
            "ANNEE": pd.Timestamp(d).year,
        }
        for c in extra_share_cols:
            next_row[c] = hist[c].iloc[-1] if c in hist.columns else 0.0

        feat_dict = one_step_share_features(hist, d, "PART_VP", extra_share_cols)
        feat_df = pd.DataFrame([feat_dict])[share_features]
        part_vp_pred = float(np.clip(share_model.predict(feat_df)[0], 0.05, 0.95))
        next_row["PART_VP"] = part_vp_pred
        next_row["PART_VU"] = 1.0 - part_vp_pred
        next_row["VP"] = next_row["TOTAL"] * part_vp_pred
        next_row["VU"] = next_row["TOTAL"] - next_row["VP"]
        hist = pd.concat([hist, pd.DataFrame([next_row])], ignore_index=True)

    future_slice = hist[hist["Date"].isin(future_dates)].copy()

    # Model 3: ARTES brands combined (share over total).
    print("3) Forecast ARTES brands (Renault + Dacia + Nissan)...")
    arts_sup, arts_features, arts_target = make_share_supervised(df_month, "ARTES_SHARE", extra_share_cols)
    arts_train = arts_sup[arts_sup["Date"] < pd.Timestamp("2025-01-01")].copy()
    arts_test = arts_sup[(arts_sup["Date"] >= pd.Timestamp("2025-01-01")) & (arts_sup["Date"] <= pd.Timestamp("2025-12-01"))].copy()

    arts_model = build_ml_regressor()
    if not arts_train.empty:
        arts_model.fit(arts_train[arts_features], arts_train[arts_target])

    if not arts_test.empty:
        arts_share_pred = np.clip(arts_model.predict(arts_test[arts_features]), 0.0, 0.8)
        arts_vol_pred = arts_share_pred * arts_test["TOTAL"].values
        model_results.append(evaluate_model("XGB_ARTES_SHARE", arts_test["ARTES_SHARE"], arts_share_pred))
        model_results.append(evaluate_model("ARTES_VOL", arts_test["ARTES_VOL"], arts_vol_pred))

    hist_artes = df_month.copy().reset_index(drop=True)
    for d, total_pred in zip(df_future["Date"], df_future["TOTAL_PRED"]):
        if d not in set(hist_artes["Date"]):
            row = {
                "Date": d,
                "TOTAL": float(total_pred),
                "RAMADAN_DAYS": month_ramadan_days(d),
                "EST_RAMADAN": int(month_ramadan_days(d) > 0),
                "MOIS": pd.Timestamp(d).month,
                "ANNEE": pd.Timestamp(d).year,
                "PART_VP": float(future_slice.loc[future_slice["Date"] == d, "PART_VP"].iloc[0]),
                "PART_VU": float(future_slice.loc[future_slice["Date"] == d, "PART_VU"].iloc[0]),
            }
            for c in extra_share_cols:
                row[c] = hist_artes[c].iloc[-1] if c in hist_artes.columns else 0.0
            hist_artes = pd.concat([hist_artes, pd.DataFrame([row])], ignore_index=True)

        feat_dict = one_step_share_features(hist_artes, d, "ARTES_SHARE", extra_share_cols)
        feat_df = pd.DataFrame([feat_dict])[arts_features]
        share_pred = float(np.clip(arts_model.predict(feat_df)[0], 0.0, 0.8))
        mask = hist_artes["Date"] == d
        hist_artes.loc[mask, "ARTES_SHARE"] = share_pred
        hist_artes.loc[mask, "ARTES_VOL"] = hist_artes.loc[mask, "TOTAL"] * share_pred

    future_artes = hist_artes[hist_artes["Date"].isin(future_dates)][["Date", "ARTES_SHARE", "ARTES_VOL"]].copy()

    # Model 4: city watch output (monitoring, not a strict forecast model).
    print("4) Build city watch output...")
    city_watch = build_city_watch(df_raw)

    # Consolidated S1 2026 output.
    forecast = future_slice[["Date", "TOTAL", "VP", "VU", "PART_VP", "PART_VU", "RAMADAN_DAYS"]].copy()
    forecast = forecast.rename(columns={"TOTAL": "PREV_TOTAL_MARCHE", "VP": "PREV_VP", "VU": "PREV_VU"})
    forecast = forecast.merge(future_artes, on="Date", how="left")
    forecast = forecast.rename(columns={"ARTES_SHARE": "PREV_PART_ARTES", "ARTES_VOL": "PREV_VOL_ARTES"})

    # Save outputs.
    out_forecast = os.path.join(project_root, "step6_forecast_s1_2026.csv")
    out_metrics = os.path.join(project_root, "step6_metrics_summary.csv")
    out_backtest = os.path.join(project_root, "step6_backtest_2025_total.csv")
    out_city = os.path.join(project_root, "step6_city_watch_s1_2025_vs_2024.csv")
    out_artes_hist = os.path.join(project_root, "step6_artes_monthly_history.csv")

    forecast.to_csv(out_forecast, index=False)
    pd.DataFrame(
        [{"name": m.name, "mae": m.mae, "rmse": m.rmse, "smape": m.smape} for m in model_results]
    ).to_csv(out_metrics, index=False)

    backtest_df = pd.DataFrame({
        "Date": test["Date"],
        "ACTUAL_TOTAL": test["TOTAL"].values,
        "PRED_ENSEMBLE_TOTAL": np.asarray(ensemble_test),
    })
    if sarimax_pred is not None:
        backtest_df["PRED_SARIMAX_TOTAL"] = sarimax_pred.values
    if prophet_pred is not None:
        backtest_df["PRED_PROPHET_TOTAL"] = prophet_pred.values
    backtest_df.to_csv(out_backtest, index=False)

    city_watch.to_csv(out_city, index=False)
    df_month[["Date", "ARTES_VOL", "ARTES_SHARE", "TOTAL"]].to_csv(out_artes_hist, index=False)

    # Visual summary.
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(df_month["Date"], df_month["TOTAL"], label="Historique total", color="steelblue")
    axes[0, 0].plot(backtest_df["Date"], backtest_df["PRED_ENSEMBLE_TOTAL"], label="Backtest 2025 (ensemble)", color="darkorange")
    axes[0, 0].plot(forecast["Date"], forecast["PREV_TOTAL_MARCHE"], label="Prevision S1 2026", color="crimson")
    axes[0, 0].set_title("Marche total mensuel")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(df_month["Date"], df_month["PART_VP"], label="Part VP historique", color="green")
    axes[0, 1].plot(forecast["Date"], forecast["PART_VP"], label="Part VP prevision", color="black")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Decomposition VP/VU")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(forecast["Date"].dt.strftime("%Y-%m"), forecast["PREV_VOL_ARTES"], color="purple")
    axes[1, 0].set_title("Volume previsionnel ARTES (Renault+Dacia+Nissan)")
    axes[1, 0].tick_params(axis="x", rotation=30)

    top_city = city_watch.head(10).copy()
    axes[1, 1].barh(top_city["VILLE"].astype(str), top_city["VENTES_S1_2025"], color="teal")
    axes[1, 1].set_title("Top 10 villes - Ventes S1 2025")

    plt.suptitle("Step 6 - Forecasting Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(project_root, "12_Step6_Forecasts_S1_2026.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Console report.
    print("\n" + "=" * 70)
    print("STEP 6 SUMMARY")
    print("=" * 70)
    print(f"Models available: statsmodels={HAS_STATSMODELS}, prophet={HAS_PROPHET}, xgboost={HAS_XGBOOST}")
    print("Backtest (2025) metrics:")
    for m in model_results:
        print(f"  - {m.name:18s} | MAE={m.mae:8.2f} | RMSE={m.rmse:8.2f} | sMAPE={m.smape:6.2f}%")

    print("\nS1 2026 forecast (monthly totals):")
    for _, row in forecast.iterrows():
        print(
            f"  {row['Date'].strftime('%Y-%m')}: TOTAL={row['PREV_TOTAL_MARCHE']:.0f}, "
            f"VP={row['PREV_VP']:.0f}, VU={row['PREV_VU']:.0f}, "
            f"ARTES={row['PREV_VOL_ARTES']:.0f} ({100 * row['PREV_PART_ARTES']:.1f}%)"
        )

    print("\nSaved files:")
    for f in [out_forecast, out_metrics, out_backtest, out_city, out_artes_hist, plot_path]:
        print(f"  - {os.path.basename(f)}")

    print("\nSTEP 6 COMPLETE")


if __name__ == "__main__":
    main()
