import pandas as pd
import streamlit as st

from components import auth, charts, data_loader, models, header

st.set_page_config(
    page_title="ARTES — Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

auth.apply_global_styles()


def _latest_month(df: pd.DataFrame, col: str) -> pd.Series:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    return work.sort_values("Date").iloc[-1][col]


def main():
    # Check authentication
    auth.check_session_timeout()
    
    if not auth.is_authenticated():
        # Hide sidebar for login page
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] { display: none; }
                .stMainBlockContainer { padding-top: 2rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        auth.render_login_page()
        st.stop()
    
    # User is authenticated; show the dashboard with header
    header.render_header()
    header.render_sidebar_minimal()

    st.title("🏠 Accueil")
    st.caption("Vue synthétique du système de prévision des ventes ARTES")

    with st.spinner("Chargement des indicateurs..."):
        forecast = data_loader.load_forecast()
        metrics = models.model_metrics_table()
        backtest = data_loader.load_backtest()
        city = data_loader.load_city_watch()

    latest_total = float(forecast["PREV_TOTAL_MARCHE"].sum())
    latest_vp = float(forecast["PREV_VP"].sum())
    latest_vu = float(forecast["PREV_VU"].sum())
    top_city = city.sort_values(["VENTES_S1_2025", "CROISSANCE_%"], ascending=[False, False]).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(auth.metric_card_html(auth.format_k(latest_total), "Prévision H1 2026 (total)", "+ vs scénario central", "primary"), unsafe_allow_html=True)
    c2.markdown(auth.metric_card_html(auth.format_k(latest_vp), "Prévision VP H1 2026", "VP structurants", "orange"), unsafe_allow_html=True)
    c3.markdown(auth.metric_card_html(auth.format_k(latest_vu), "Prévision VU H1 2026", "VU en soutien", "green"), unsafe_allow_html=True)
    c4.markdown(auth.metric_card_html(auth.format_k(top_city["VENTES_S1_2025"]), f"Ville leader : {top_city['VILLE']}", f"{auth.format_pct(top_city['CROISSANCE_%'] if pd.notna(top_city['CROISSANCE_%']) else 0)} vs S1 2024", "red"), unsafe_allow_html=True)

    st.markdown("### Dernière performance des modèles")
    metric_cols = st.columns(4)
    for col, (_, row) in zip(metric_cols, metrics.head(4).iterrows()):
        col.markdown(auth.metric_card_html(auth.format_pct(row["MAPE test"]), row["Modèle"], f"RMSE {auth.format_k(row['RMSE'])}", "primary"), unsafe_allow_html=True)

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("### Courbe de prévision")
        st.plotly_chart(
            charts.forecast_time_series(
                forecast,
                date_col="Date",
                history_col=None,
                val_col=None,
                test_col=None,
                forecast_col="PREV_TOTAL_MARCHE",
                confidence_band=("CI95_LOWER", "CI95_UPPER") if {"CI95_LOWER", "CI95_UPPER"}.issubset(forecast.columns) else None,
                title="Prévisions S1 2026",
                subtitle="Vue synthétique sur l'horizon principal",
            ),
            use_container_width=True,
            config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}},
        )
    with right:
        st.markdown("### Backtest 2025")
        st.dataframe(backtest.tail(6), use_container_width=True, height=260)
        st.markdown("### Villes à surveiller")
        st.dataframe(city.head(8), use_container_width=True, height=260)

    auth.footer()


if __name__ == "__main__":
    main()
