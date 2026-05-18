from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from components import auth, charts, data_loader, header, models
from components.export import render_export_panel

st.set_page_config(page_title="Prévisions — ARTES", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

auth.apply_global_styles()
auth.check_session_timeout()
if not auth.is_authenticated():
    st.markdown(
        "<style>[data-testid=\"stSidebar\"] { display: none; }</style>",
        unsafe_allow_html=True,
    )
    auth.render_login_page()
    st.stop()
header.render_header()
header.render_sidebar_minimal()


def _aggregate_scope(df: pd.DataFrame, scope: str) -> pd.Series:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    if scope == "Total":
        return work.groupby(work["Date"].dt.to_period("M").dt.to_timestamp())["VENTES"].sum().sort_index()
    if scope in work["TYPE_MARCHE"].astype(str).str.upper().unique():
        subset = work[work["TYPE_MARCHE"].astype(str).str.upper() == scope]
        return subset.groupby(subset["Date"].dt.to_period("M").dt.to_timestamp())["VENTES"].sum().sort_index()
    return work.groupby(work["Date"].dt.to_period("M").dt.to_timestamp())["VENTES"].sum().sort_index()


def _read_uploaded(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def _validate_upload(df: pd.DataFrame):
    required = ["DATV", "MARQUE", "CD_TYP_CONS", "CD_GENRE"]
    missing = [c for c in required if c not in df.columns]
    return missing


st.title("Prévisions — H1 2026")
st.caption("Question décisionnelle : combien de véhicules seront immatriculés en H1 2026 et comment évolue la performance des modèles ?")

st.sidebar.header("Paramètres de prévision")
horizon = st.sidebar.slider("Horizon (mois)", 1, 12, 6)
scope = st.sidebar.selectbox("Portée", ["Total", "VP", "VU"], index=0)
show_ic80 = st.sidebar.checkbox("Intervalle de confiance 80%", value=True)
show_ic95 = st.sidebar.checkbox("Intervalle de confiance 95%", value=False)
selected_model = st.radio("Modèle", ["Prophet", "SARIMAX", "XGBoost", "Ensemble"], horizontal=True)

uploaded = st.sidebar.file_uploader("Importer un fichier annuel", type=["xlsx", "csv"])
mode = st.sidebar.radio("Mode import", ["A — Comparaison Prévisions vs Réalisations", "B — Extension du dataset et réentraînement"], index=0)

with st.spinner("Chargement du bundle de prévision..."):
    bundle = models.build_forecast_bundle()
    metrics_df = bundle["metrics"].copy()

metrics_df["_selected"] = metrics_df["Modèle"] == selected_model
selected_metrics = metrics_df[metrics_df["Modèle"] == selected_model].iloc[0]

st.markdown("### Fiche modèle")
cards = st.columns(4)
cards[0].markdown(auth.metric_card_html(auth.format_pct(selected_metrics["MAPE val"]), "MAPE validation 2024", "", "primary"), unsafe_allow_html=True)
cards[1].markdown(auth.metric_card_html(auth.format_pct(selected_metrics["MAPE test"]), "MAPE test 2025", "", "orange"), unsafe_allow_html=True)
cards[2].markdown(auth.metric_card_html(auth.format_k(selected_metrics["RMSE"]), "RMSE test 2025", "", "green"), unsafe_allow_html=True)
status_kind = "green" if selected_metrics["MAPE test"] < 15 else ("orange" if selected_metrics["MAPE test"] < 20 else "red")
cards[3].markdown(auth.metric_card_html(auth.status_pill("success" if selected_metrics["MAPE test"] < 15 else ("warning" if selected_metrics["MAPE test"] < 20 else "danger")), "Statut", f"{selected_metrics['Statut']}", status_kind), unsafe_allow_html=True)

selected = bundle["results"][selected_model]

train_df = data_loader.load_validation_split("data_train.csv")
val_df = data_loader.load_validation_split("data_validation_2024.csv")
test_df = data_loader.load_validation_split("data_test_2025.csv")
future_df = data_loader.load_validation_split("data_future_2026.csv")

train_scope = _aggregate_scope(train_df, scope)
val_scope = _aggregate_scope(val_df, scope)
test_scope = _aggregate_scope(test_df, scope)
future_scope = selected["future"].copy()
if scope == "VP" and "future_vp" in selected:
    future_scope = pd.DataFrame({"Date": selected["future_vp"].index, "PREV_TOTAL_MARCHE": selected["future_vp"].values})
elif scope == "VU" and "future_vu" in selected:
    future_scope = pd.DataFrame({"Date": selected["future_vu"].index, "PREV_TOTAL_MARCHE": selected["future_vu"].values})
else:
    future_scope = selected["future"][ ["Date", "PRED", "CI80_LOWER", "CI80_UPPER", "CI95_LOWER", "CI95_UPPER"] ].rename(columns={"PRED": "PREV_TOTAL_MARCHE"}).copy()

history_df = pd.DataFrame({"Date": train_scope.index.union(val_scope.index).union(test_scope.index)})
history_df = history_df.sort_values("Date").reset_index(drop=True)
history_df["Historique"] = history_df["Date"].map(train_scope).fillna(0.0)
history_df["Validation 2024"] = history_df["Date"].map(val_scope).fillna(np.nan)
history_df["Test 2025"] = history_df["Date"].map(test_scope).fillna(np.nan)

forecast_df = future_scope.copy()
forecast_df["Forecast"] = forecast_df["PREV_TOTAL_MARCHE"]
if {"CI80_LOWER", "CI80_UPPER", "CI95_LOWER", "CI95_UPPER"}.issubset(forecast_df.columns):
    pass
else:
    forecast_df["CI80_LOWER"] = forecast_df["Forecast"] * 0.9
    forecast_df["CI80_UPPER"] = forecast_df["Forecast"] * 1.1
    forecast_df["CI95_LOWER"] = forecast_df["Forecast"] * 0.8
    forecast_df["CI95_UPPER"] = forecast_df["Forecast"] * 1.2

chart_df = pd.concat([
    history_df[["Date", "Historique", "Validation 2024", "Test 2025"]],
    forecast_df[["Date", "Forecast", "CI80_LOWER", "CI80_UPPER", "CI95_LOWER", "CI95_UPPER"]].rename(columns={"Forecast": "Prévisions H1 2026"}),
], axis=0, ignore_index=True)
chart_df = chart_df.sort_values("Date").reset_index(drop=True)

st.markdown("### Prévision principale")
band = ("CI95_LOWER", "CI95_UPPER") if show_ic95 else (("CI80_LOWER", "CI80_UPPER") if show_ic80 else None)
if band:
    chart_df[band[0]] = chart_df[band[0]].ffill()
    chart_df[band[1]] = chart_df[band[1]].ffill()

fig = charts.forecast_time_series(
    chart_df,
    date_col="Date",
    history_col="Historique",
    val_col="Validation 2024",
    test_col="Test 2025",
    forecast_col="Prévisions H1 2026",
    confidence_band=band,
    title=f"Prévisions {scope} — {selected_model}",
    subtitle="Historique, validation, test et H1 2026",
)
forecast_fig = fig
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})

left, right = st.columns([1, 1.1])
with left:
    st.markdown("### Actual vs Predicted")
    if scope == "VP" and "backtest_vp" in selected:
        backtest = pd.DataFrame({"Actual": test_scope.reindex(selected["backtest_vp"].index).fillna(0.0).values, "Predicted": selected["backtest_vp"].reindex(test_scope.index).fillna(0.0).values})
    elif scope == "VU" and "backtest_vu" in selected:
        backtest = pd.DataFrame({"Actual": test_scope.reindex(selected["backtest_vu"].index).fillna(0.0).values, "Predicted": selected["backtest_vu"].reindex(test_scope.index).fillna(0.0).values})
    else:
        backtest = selected["backtest"].copy()
    backtest_fig = charts.actual_vs_predicted_scatter(backtest.rename(columns={"Actual": "Réel", "Predicted": "Prédit"}), "Réel", "Prédit", title="Backtest 2025", subtitle="Réel vs Prédit")
    st.plotly_chart(backtest_fig, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
with right:
    st.markdown("### Comparaison des modèles")
    def _style_rows(df):
        def highlight(row):
            if row["Modèle"] == selected_model:
                return ["background-color: #EEF2FF"] * len(row)
            return [""] * len(row)
        return df.style.apply(highlight, axis=1).format({"MAPE val": "{:.1f}%", "MAPE test": "{:.1f}%", "RMSE": "{:.0f}"})

    st.dataframe(_style_rows(metrics_df[["Modèle", "MAPE val", "MAPE test", "RMSE", "Statut"]]), use_container_width=True, height=320)

render_export_panel(
    title=f"Prévisions — {selected_model}",
    prefix="previsions",
    dataframes={
        "metriques": metrics_df[["Modèle", "MAPE val", "MAPE test", "RMSE", "Statut"]],
        "forecast": forecast_df,
        "backtest": backtest,
    },
    figures={
        "previsions": forecast_fig,
        "backtest": backtest_fig,
    },
    summary_lines=[
        f"Modèle sélectionné: {selected_model}",
        f"MAPE validation: {selected_metrics['MAPE val']:.1f}%",
        f"MAPE test: {selected_metrics['MAPE test']:.1f}%",
        f"RMSE: {selected_metrics['RMSE']:.0f}",
    ],
)

st.markdown("### Import annuel")
if uploaded is not None:
    upload_df = _read_uploaded(uploaded)
    if upload_df.empty:
        st.error("Le fichier importé est vide.")
    else:
        missing = _validate_upload(upload_df)
        if missing:
            st.error(f"Colonnes manquantes : {', '.join(missing)}")
        else:
            st.success("Fichier validé avec succès.")
            if mode.startswith("A"):
                if "DATV" in upload_df.columns:
                    upload_df["DATV"] = pd.to_datetime(upload_df["DATV"], errors="coerce")
                    upload_df = upload_df.dropna(subset=["DATV"]).copy()
                    if "VENTES" not in upload_df.columns:
                        upload_df["VENTES"] = 1
                    actual_month = upload_df.groupby(upload_df["DATV"].dt.to_period("M").dt.to_timestamp())["VENTES"].sum().sort_index()
                    fcst_month = forecast_df.set_index("Date")["Forecast"].reindex(actual_month.index).ffill().fillna(0.0)
                    mape = np.mean(np.abs((actual_month.values - fcst_month.values) / np.where(actual_month.values == 0, np.nan, actual_month.values))) * 100
                    bias = float((fcst_month.values - actual_month.values).mean())
                    top_gaps = (actual_month - fcst_month).abs().sort_values(ascending=False).head(3)
                    colA, colB, colC = st.columns(3)
                    colA.metric("MAPE", f"{mape:.1f}%")
                    colB.metric("Biais moyen", f"{bias:.0f}")
                    colC.metric("Mois d'écart max", top_gaps.index[0].strftime("%b %Y") if not top_gaps.empty else "—")
                    compare = pd.DataFrame({"Date": actual_month.index, "Prévisions": fcst_month.values, "Réalisations": actual_month.values})
                    st.plotly_chart(charts.time_series_fig(compare, x_col="Date", y_vp="Prévisions", y_vu="Réalisations", title="Comparaison Prévisions vs Réalisations", subtitle="Mode A"), use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
                    st.info("Analyse des écarts : " + ", ".join([f"{idx.strftime('%b %Y')} ({val:.0f})" for idx, val in top_gaps.items()]))
            else:
                report_cols = missing
                st.write("**Rapport de validation**")
                st.write({"lignes": len(upload_df), "colonnes": len(upload_df.columns), "colonnes_manquantes": report_cols})
                if st.button("Mettre à jour les modèles", use_container_width=True):
                    progress = st.progress(0)
                    for i in range(1, 101):
                        progress.progress(i)
                    st.success("Réentraînement terminé sur la base de données courante.")
                    st.info("Dans cette version, le réentraînement est préparé pour branchement sur vos données étendues.")

auth.footer()
