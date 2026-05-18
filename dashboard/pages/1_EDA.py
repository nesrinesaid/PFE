from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import auth, charts, data_loader, header, models
from components.export import render_export_panel

st.set_page_config(page_title="Analyse Exploratoire — ARTES", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

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


def _clean_frame() -> pd.DataFrame:

    # Debug: Page is loading
    st.write("DEBUG: EDA page is loading...")

    df = data_loader.load_monthly_prepared().copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "VENTES" not in df.columns:
        df["VENTES"] = 1
    else:
        df["VENTES"] = pd.to_numeric(df["VENTES"], errors="coerce").fillna(0.0)
    if "TYPE_MARCHE" not in df.columns:
        df["TYPE_MARCHE"] = "VP"
    df["TYPE_MARCHE"] = df["TYPE_MARCHE"].astype(str).str.upper().replace({"MARCHÉ VP": "VP", "MARCHE VP": "VP", "MARCHÉ VU": "VU", "MARCHE VU": "VU"})
    df["ANNEE"] = df["Date"].dt.year
    df["MOIS"] = df["Date"].dt.month
    df["MONTH"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df.dropna(subset=["Date"]).copy()


def _filter_data(df: pd.DataFrame, years, market, continent) -> pd.DataFrame:
    work = df[df["ANNEE"].isin(years)].copy()
    if market != "Tous" and "TYPE_MARCHE" in work.columns:
        work = work[work["TYPE_MARCHE"] == market].copy()
    if continent != "Tous" and "CONTINENT" in work.columns:
        work = work[work["CONTINENT"].astype(str).str.upper() == continent.upper()].copy()
    return work


df = _clean_frame()

st.title("Analyse Exploratoire — Structure et évolution du marché")
st.caption("Question décisionnelle : Quelle est la structure et l'évolution du marché automobile tunisien ?")

years = st.sidebar.multiselect("Années", sorted(df["ANNEE"].dropna().astype(int).unique().tolist()), default=sorted(df["ANNEE"].dropna().astype(int).unique().tolist()))
market = st.sidebar.selectbox("Type de marché", ["Tous", "VP", "VU"], index=0)
continent = st.sidebar.selectbox("Continent", ["Tous", "Afrique", "Amérique", "Asie", "Europe"], index=0)

f = _filter_data(df, years, market, continent)

monthly = f.groupby(["MONTH", "TYPE_MARCHE"], as_index=False)["VENTES"].sum().pivot(index="MONTH", columns="TYPE_MARCHE", values="VENTES").fillna(0.0).reset_index()
if "VP" not in monthly.columns:
    monthly["VP"] = 0.0
if "VU" not in monthly.columns:
    monthly["VU"] = 0.0

monthly_total = f.groupby("MONTH", as_index=False)["VENTES"].sum().sort_values("MONTH")
annual = f.groupby("ANNEE", as_index=False).agg(VP=("TYPE_MARCHE", lambda s: int((s == "VP").sum())), VU=("TYPE_MARCHE", lambda s: int((s == "VU").sum())))
top_brand = f.groupby("MARQUE", as_index=False)["VENTES"].sum().sort_values("VENTES", ascending=False) if "MARQUE" in f.columns else pd.DataFrame(columns=["MARQUE", "VENTES"])
segment = f.groupby("SEGMENT", as_index=False)["VENTES"].sum().sort_values("VENTES", ascending=False) if "SEGMENT" in f.columns else pd.DataFrame(columns=["SEGMENT", "VENTES"])
energy = f.groupby(["MONTH", "ENERGIE"], as_index=False)["VENTES"].sum() if "ENERGIE" in f.columns else pd.DataFrame()
city = f.groupby("VILLE", as_index=False)["VENTES"].sum().sort_values("VENTES", ascending=False) if "VILLE" in f.columns else pd.DataFrame(columns=["VILLE", "VENTES"])
continent_share = f.groupby("CONTINENT", as_index=False)["VENTES"].sum().sort_values("VENTES", ascending=False) if "CONTINENT" in f.columns else pd.DataFrame(columns=["CONTINENT", "VENTES"])
color_col = "CONTINENT" if "CONTINENT" in f.columns else None
top_brand2 = f.groupby(["MARQUE"] + (["CONTINENT"] if color_col else []), as_index=False)["VENTES"].sum().sort_values("VENTES", ascending=False) if not top_brand.empty else pd.DataFrame(columns=["MARQUE", "VENTES"])
energy_p = energy.pivot_table(index="MONTH", columns="ENERGIE", values="VENTES", aggfunc="sum", fill_value=0).reset_index() if not energy.empty else pd.DataFrame()
keep = [c for c in energy_p.columns if c != "MONTH"][:4] if not energy_p.empty else []
city15 = city.head(15).copy() if not city.empty else pd.DataFrame(columns=["VILLE", "VENTES"])

total_reg = float(f["VENTES"].sum())
brands = int(f["MARQUE"].nunique()) if "MARQUE" in f.columns else 0
peak_idx = monthly_total["VENTES"].idxmax() if not monthly_total.empty else None
peak_month = monthly_total.loc[peak_idx, "MONTH"] if peak_idx is not None and peak_idx in monthly_total.index else None
peak_value = monthly_total.loc[peak_idx, "VENTES"] if peak_idx is not None and peak_idx in monthly_total.index else 0
vp = float((f["TYPE_MARCHE"] == "VP").sum())
vu = float((f["TYPE_MARCHE"] == "VU").sum())
split = vp + vu if vp + vu > 0 else 1

if 2024 in years and 2025 in years:
    split_2024 = f[f["ANNEE"] == 2024]["TYPE_MARCHE"].value_counts(normalize=True)
    split_2025 = f[f["ANNEE"] == 2025]["TYPE_MARCHE"].value_counts(normalize=True)
    split_delta = (split_2025.get("VP", 0) - split_2024.get("VP", 0)) * 100
else:
    split_delta = 0.0

fig_market = charts.time_series_fig(monthly, x_col="MONTH", y_vp="VP", y_vu="VU")
year_pivot = f.groupby(["ANNEE", "TYPE_MARCHE"], as_index=False)["VENTES"].sum().pivot(index="ANNEE", columns="TYPE_MARCHE", values="VENTES").fillna(0).reset_index()

# Ensure VP and VU columns exist even if they're empty
for col in ["VP", "VU"]:
    if col not in year_pivot.columns:
        year_pivot[col] = 0.0

fig_year = charts.grouped_year_bars(year_pivot, year_col="ANNEE", series_cols=("VP", "VU"), title="Ventes par année", subtitle="VP vs VU")
fig_heat = charts.monthly_heatmap(f, value_col="VENTES", year_col="ANNEE", month_col="MOIS", title="Saisonnalité mensuelle", subtitle="Année × Mois")
fig_brand = charts.horizontal_top_bar(top_brand2, "MARQUE", "VENTES", color_col=color_col, top_n=15, title="Top 15 marques", subtitle="Classement par volume décroissant") if not top_brand.empty else None
fig_segment = charts.donut_chart(segment, "SEGMENT", "VENTES", title="Répartition par segment", subtitle="Maximum 6 tranches") if not segment.empty else None
fig_energy = charts.stacked_area_chart(energy_p.rename(columns={c: c for c in keep}), x_col="MONTH", series_cols=keep, title="Distribution de l'énergie", subtitle="Essence / Gasoil / Hybride / Électrique") if not energy.empty else None
fig_city = charts.horizontal_top_bar(city15, "VILLE", "VENTES", title="Répartition géographique des immatriculations", subtitle="Top 15 villes tunisiennes") if not city.empty else None
fig_continent = charts.donut_chart(continent_share, "CONTINENT", "VENTES", title="Répartition par continent", subtitle="Exposition géographique") if not continent_share.empty else None

cards = st.columns(4)
cards[0].markdown(auth.metric_card_html(auth.format_k(total_reg), "Total immatriculations", f"{auth.format_pct((total_reg / max(1.0, df['VENTES'].sum())) * 100 - 100)} vs base" if total_reg else "", "primary"), unsafe_allow_html=True)
cards[1].markdown(auth.metric_card_html(str(brands), "Marques distinctes", "+ nouvelles catégories" if brands else "", "orange"), unsafe_allow_html=True)
cards[2].markdown(auth.metric_card_html(f"{peak_month.strftime('%b %Y') if peak_month is not None else '—'}", "Mois le plus fort", auth.format_k(peak_value), "green"), unsafe_allow_html=True)
cards[3].markdown(auth.metric_card_html(f"{int(round(100 * vp / split))}% / {int(round(100 * vu / split))}%", "Part VP / VU", f"{auth.format_delta(split_delta)} VP vs 2024", "primary"), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Marché", "Structure", "Géographie"])

with tab1:
    left, right = st.columns([1.8, 1])
    with left:
        st.plotly_chart(fig_market, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
    with right:
        st.plotly_chart(fig_year, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})

with tab2:
    left, right = st.columns([1.3, 1])
    with left:
        if not top_brand.empty:
            st.plotly_chart(fig_brand, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
        else:
            st.info("La colonne MARQUE est indisponible dans ce filtrage.")
    with right:
        if not segment.empty:
            st.plotly_chart(fig_segment, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
        if not energy.empty:
            st.plotly_chart(fig_energy, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})

with tab3:
    left, right = st.columns([1.3, 1])
    with left:
        if not city.empty:
            st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
    with right:
        if not continent_share.empty:
            st.plotly_chart(fig_continent, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})

render_export_panel(
    title="Analyse Exploratoire — ARTES",
    prefix="eda",
    dataframes={
        "kpis": pd.DataFrame([
            {"indicateur": "Total immatriculations", "valeur": total_reg},
            {"indicateur": "Marques distinctes", "valeur": brands},
            {"indicateur": "Mois le plus fort", "valeur": peak_month.strftime("%b %Y") if peak_month is not None else "—"},
        ]),
        "donnees_filtrees": f,
    },
    figures={
        "g1_evolution_mensuelle": fig_market,
        "g2_saisonnalite": fig_heat,
        "g3_ventes_par_annee": fig_year,
        "g4_top_marques": fig_brand,
        "g5_segment": fig_segment,
        "g6_energie": fig_energy,
        "g7_villes": fig_city,
        "g8_continent": fig_continent,
    },
    summary_lines=[
        f"Total immatriculations filtrées: {auth.format_k(total_reg)}",
        f"Marques distinctes: {brands}",
        f"Mois le plus fort: {peak_month.strftime('%b %Y') if peak_month is not None else '—'}",
    ],
)

auth.footer()
