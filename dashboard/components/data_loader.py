from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]


def project_path(*parts) -> Path:
    return ROOT_DIR.joinpath(*parts)


def find_data_file(name: str):
    candidates = [
        project_path(name),
        project_path("data", name),
        project_path("dashboard", "data", name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: str, parse_dates=None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_data(show_spinner=False)
def load_data_file(name: str, parse_dates=None) -> pd.DataFrame:
    path = find_data_file(name)
    if not path:
        raise FileNotFoundError(name)
    return load_csv(path, parse_dates=parse_dates)


@st.cache_data(show_spinner=False)
def load_monthly_prepared() -> pd.DataFrame:
    return load_data_file("data_prepared_final.csv", parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_cleaned_enriched() -> pd.DataFrame:
    return load_data_file("data_cleaned_enriched.csv", parse_dates=["DATV", "DATE_MEC"])


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    path = find_data_file("step6_forecast_s1_2026.csv") or find_data_file("step6_forecast_s1_2026_baseline.csv")
    if not path:
        raise FileNotFoundError("step6_forecast_s1_2026.csv")
    return load_csv(path, parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_metrics_summary() -> pd.DataFrame:
    return load_data_file("step6_metrics_summary.csv")


@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    return load_data_file("step6_backtest_2025_total.csv", parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_city_watch() -> pd.DataFrame:
    return load_data_file("step6_city_watch_s1_2025_vs_2024.csv")


@st.cache_data(show_spinner=False)
def load_segment_forecast(scenario: str = "baseline") -> pd.DataFrame:
    path = find_data_file(f"step6_forecast_s1_2026_by_segment_{scenario}.csv") or find_data_file("step6_forecast_s1_2026_by_segment.csv")
    if not path:
        raise FileNotFoundError("segment forecast file")
    return load_csv(path, parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_sous_segment_forecast(scenario: str = "baseline") -> pd.DataFrame:
    path = find_data_file(f"step6_forecast_s1_2026_by_sous_segment_{scenario}.csv") or find_data_file("step6_forecast_s1_2026_by_sous_segment.csv")
    if not path:
        raise FileNotFoundError("sous-segment forecast file")
    return load_csv(path, parse_dates=["Date"])


@st.cache_data(show_spinner=False)
def load_validation_split(name: str) -> pd.DataFrame:
    return load_data_file(name, parse_dates=["Date"])

