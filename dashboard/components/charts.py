from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "page_bg": "#E9EBF7",
    "sidebar_bg": "#FFFFFF",
    "card_bg": "#FFFFFF",
    "panel_bg": "#F4F5FB",
    "primary": "#4A69A9",
    "primary_light": "#C9EAFF",
    "primary_dark": "#2E4A80",
    "accent_orange": "#FF944B",
    "accent_blue": "#0054CF",
    "accent_green": "#2ECC8F",
    "accent_red": "#E84855",
    "accent_purple": "#9B59B6",
    "text_primary": "#1E2A45",
    "text_secondary": "#6B7490",
    "text_muted": "#9BA3B8",
    "chart_grid": "#E2E5F0",
    "ramadan_zone": "#EEF0FA",
    "vp_color": "#0054CF",
    "vu_color": "#FF944B",
    "forecast_color": "#E84855",
    "history_color": "#4A69A9",
    "val_color": "#2ECC8F",
    "test_color": "#9B59B6",
    "success": "#2ECC8F",
    "warning": "#F39C12",
    "danger": "#E84855",
    "info": "#4A69A9",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", color="#1E2A45"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="#E2E5F0",
        gridwidth=0.5,
        linecolor="#E2E5F0",
        tickfont=dict(color="#6B7490", size=11),
        title_font=dict(color="#6B7490", size=12),
    ),
    yaxis=dict(
        gridcolor="#E2E5F0",
        gridwidth=0.5,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color="#6B7490", size=11),
        title_font=dict(color="#6B7490", size=12),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(color="#6B7490", size=11),
    ),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        bordercolor="#E2E5F0",
        font=dict(family="Inter, sans-serif", color="#1E2A45", size=12),
    ),
)

RAMADAN_RANGES = [
    ("2019-05-05", "2019-06-04"),
    ("2021-04-13", "2021-05-13"),
    ("2022-04-02", "2022-05-02"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-30"),
    ("2026-02-18", "2026-03-19"),
]


def apply_plotly_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def _add_plotly_title(fig: go.Figure, title: str, subtitle: str = ""):
    fig.update_layout(title=dict(text=f"{title}<br><span style='font-size:12px;color:#9BA3B8'>{subtitle}</span>", x=0.0, xanchor="left"))


def add_ramadan_zones(fig: go.Figure, x_ref="x", annotate: bool = True):
    for start, end in RAMADAN_RANGES:
        fig.add_vrect(x0=start, x1=end, fillcolor=COLORS["ramadan_zone"], opacity=0.75, line_width=0, layer="below")
    if annotate:
        fig.add_annotation(
            x=RAMADAN_RANGES[0][0], y=1.02, yref="paper", xref="x",
            text="Ramadan", showarrow=False, font=dict(color=COLORS["text_muted"], size=11),
            bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0)",
        )
    return fig


def _format_text(value, kind="k"):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if kind == "pct":
        return f"{value:.1f}%"
    if abs(value) >= 1000:
        return f"{value/1000:.1f}K"
    return f"{value:.0f}"


def time_series_fig(df: pd.DataFrame, x_col="Date", y_vp="VP", y_vu="VU", title="Évolution mensuelle des immatriculations VP et VU", subtitle="2019 – 2025 • Données immatriculées en Tunisie"):
    fig = go.Figure()
    if y_vp in df.columns:
        fig.add_trace(go.Scatter(
            name="VP",
            x=df[x_col],
            y=df[y_vp],
            mode="lines+markers",
            line=dict(color=COLORS["vp_color"], width=2.5),
            marker=dict(size=6),
            hovertemplate="Période: %{x|%b %Y}<br>Valeur: %{y:,.0f}<br>% du total: —<extra>VP</extra>",
        ))
    if y_vu in df.columns:
        fig.add_trace(go.Scatter(
            name="VU",
            x=df[x_col],
            y=df[y_vu],
            mode="lines+markers",
            line=dict(color=COLORS["vu_color"], width=2.5),
            marker=dict(size=6),
            hovertemplate="Période: %{x|%b %Y}<br>Valeur: %{y:,.0f}<br>% du total: —<extra>VU</extra>",
        ))
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    fig.update_xaxes(type="date")
    add_ramadan_zones(fig)
    return fig


def monthly_heatmap(df: pd.DataFrame, value_col="VENTES", year_col="ANNEE", month_col="MOIS", title="Saisonnalité mensuelle", subtitle="Année × Mois"):
    pivot = df.pivot_table(index=year_col, columns=month_col, values=value_col, aggfunc="sum", fill_value=0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(int(c)) for c in pivot.columns],
            y=[str(int(i)) for i in pivot.index],
            colorscale=[[0, "#FFFFFF"], [1, COLORS["primary"]]],
            hovertemplate="Année %{y}<br>Mois %{x}<br>Valeur %{z:,.0f}<extra></extra>",
            colorbar=dict(title="Ventes"),
        )
    )
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    fig.update_xaxes(title="Mois")
    fig.update_yaxes(title="Année")
    return fig


def grouped_year_bars(df: pd.DataFrame, year_col="ANNEE", series_cols=("VP", "VU"), title="Ventes par année", subtitle="Barres groupées VP vs VU"):
    yearly = df.groupby(year_col)[list(series_cols)].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="VP", x=yearly[year_col], y=yearly[series_cols[0]], marker_color=COLORS["vp_color"], text=yearly[series_cols[0]].map(lambda v: _format_text(v)), textposition="outside"))
    fig.add_trace(go.Bar(name="VU", x=yearly[year_col], y=yearly[series_cols[1]], marker_color=COLORS["vu_color"], text=yearly[series_cols[1]].map(lambda v: _format_text(v)), textposition="outside"))
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Année")
    fig.update_yaxes(title="Nombre de véhicules")
    return fig


def horizontal_top_bar(df: pd.DataFrame, category_col: str, value_col: str, color_col: str | None = None, top_n: int = 15, title: str = "Top 15", subtitle: str = ""):
    work = df.sort_values(value_col, ascending=False).head(top_n).copy()
    colors = COLORS["primary"] if color_col is None else work[color_col].map({
        "Europe": COLORS["primary"],
        "Asie": COLORS["accent_green"],
        "Amérique": COLORS["accent_orange"],
        "Afrique": "#9B59B6",
    }).fillna(COLORS["primary"])
    fig = go.Figure(go.Bar(
        x=work[value_col],
        y=work[category_col],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#FFFFFF", width=0.5)),
        text=work[value_col].map(lambda v: _format_text(v)),
        textposition="outside",
        hovertemplate=f"{category_col}: %{{y}}<br>Valeur: %{{x:,.0f}}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    fig.update_xaxes(title="Volume")
    fig.update_yaxes(title=category_col)
    return fig


def donut_chart(df: pd.DataFrame, labels_col: str, values_col: str, title: str, subtitle: str = "", max_slices: int = 6):
    work = df[[labels_col, values_col]].copy().sort_values(values_col, ascending=False)
    if len(work) > max_slices:
        top = work.head(max_slices - 1)
        other = pd.DataFrame({labels_col: ["Autres"], values_col: [work[values_col].iloc[max_slices - 1 :].sum()]})
        work = pd.concat([top, other], ignore_index=True)
    fig = go.Figure(go.Pie(
        labels=work[labels_col],
        values=work[values_col],
        hole=0.55,
        sort=False,
        textinfo="percent+label",
        marker=dict(colors=[COLORS["primary"], COLORS["accent_orange"], COLORS["accent_green"], COLORS["accent_purple"], COLORS["accent_blue"], COLORS["danger"]][:len(work)]),
        hovertemplate="%{label}<br>Valeur: %{value:,.0f}<br>% du total: %{percent}<extra></extra>",
    ))
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    return fig


def stacked_area_chart(df: pd.DataFrame, x_col: str, series_cols: list[str], title: str, subtitle: str = ""):
    fig = go.Figure()
    palette = [COLORS["vp_color"], COLORS["vu_color"], COLORS["accent_green"], COLORS["accent_purple"]]
    for idx, col in enumerate(series_cols):
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col], name=col, mode="lines",
            line=dict(color=palette[idx % len(palette)], width=2),
            stackgroup="one", groupnorm="percent" if df[series_cols].sum(axis=1).max() <= 1.1 else None,
            hovertemplate="Période: %{x|%b %Y}<br>Valeur: %{y:,.0f}<extra>%{fullData.name}</extra>",
        ))
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    return fig


def scatter_pca(df: pd.DataFrame, x_col="PC1", y_col="PC2", cluster_col="cluster", label_col="label", title="Clusters PCA 2D", subtitle="Projection des clusters"):
    fig = go.Figure()
    clusters = sorted(df[cluster_col].dropna().unique())
    palette = [COLORS["vp_color"], COLORS["vu_color"], COLORS["accent_green"], COLORS["accent_purple"], COLORS["accent_blue"], COLORS["danger"], COLORS["warning"]]
    for idx, cluster in enumerate(clusters):
        sub = df[df[cluster_col] == cluster]
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col], mode="markers",
            name=f"Cluster {cluster}",
            text=sub[label_col] if label_col in sub.columns else None,
            marker=dict(size=10, color=palette[idx % len(palette)], opacity=0.85, line=dict(color="#FFFFFF", width=0.5)),
            hovertemplate=f"Cluster {cluster}<br>%{{text}}<br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
        ))
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    return fig


def radar_profile(df: pd.DataFrame, cluster_col: str, features: list[str], title: str, subtitle: str = ""):
    fig = go.Figure()
    palette = [COLORS["vp_color"], COLORS["vu_color"], COLORS["accent_green"], COLORS["accent_purple"], COLORS["accent_blue"], COLORS["danger"]]
    for idx, cluster in enumerate(sorted(df[cluster_col].unique())):
        subset = df[df[cluster_col] == cluster]
        values = subset[features].mean().tolist()
        values += [values[0]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features + [features[0]],
            fill="toself",
            name=f"Cluster {cluster}",
            line=dict(color=palette[idx % len(palette)]),
            opacity=0.35,
        ))
    _add_plotly_title(fig, title, subtitle)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, gridcolor=COLORS["chart_grid"])))
    apply_plotly_style(fig)
    return fig


def actual_vs_predicted_scatter(df: pd.DataFrame, actual_col: str, pred_col: str, title: str, subtitle: str = ""):
    errors = np.abs(df[actual_col] - df[pred_col]) / df[actual_col].replace(0, np.nan)
    colors = np.where(errors < 0.10, COLORS["accent_green"], np.where(errors < 0.15, COLORS["accent_orange"], COLORS["accent_red"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[actual_col], y=df[pred_col], mode="markers",
        marker=dict(color=colors, size=8, line=dict(color="#FFFFFF", width=0.5)),
        hovertemplate=f"Réel: %{{x:,.0f}}<br>Prédit: %{{y:,.0f}}<extra></extra>",
        showlegend=False,
    ))
    lim = [min(df[actual_col].min(), df[pred_col].min()), max(df[actual_col].max(), df[pred_col].max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#9BA3B8", dash="dash"), name="y=x"))
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    return fig


def forecast_time_series(df: pd.DataFrame, date_col: str, history_col: str | None, val_col: str | None, test_col: str | None, forecast_col: str, forecast_name: str = "Prévision H1 2026", confidence_band: tuple[str, str] | None = None, title: str = "Prévisions", subtitle: str = ""):
    fig = go.Figure()
    if history_col and history_col in df.columns:
        fig.add_trace(go.Scatter(x=df[date_col], y=df[history_col], mode="lines", name="Historique", line=dict(color=COLORS["history_color"], width=2.5), opacity=0.75))
    if val_col and val_col in df.columns:
        fig.add_trace(go.Scatter(x=df[date_col], y=df[val_col], mode="lines", name="Validation 2024", line=dict(color=COLORS["val_color"], width=2.5)))
    if test_col and test_col in df.columns:
        fig.add_trace(go.Scatter(x=df[date_col], y=df[test_col], mode="lines", name="Test 2025", line=dict(color=COLORS["test_color"], width=2.5)))
    if forecast_col in df.columns:
        fig.add_trace(go.Scatter(x=df[date_col], y=df[forecast_col], mode="lines", name=forecast_name, line=dict(color=COLORS["forecast_color"], width=2.5, dash="dot")))
    if confidence_band and all(c in df.columns for c in confidence_band):
        lower, upper = confidence_band
        fig.add_trace(go.Scatter(x=df[date_col], y=df[upper], line=dict(width=0), showlegend=False, hoverinfo="skip", name="Upper"))
        fig.add_trace(go.Scatter(x=df[date_col], y=df[lower], fill="tonexty", fillcolor="rgba(232,72,85,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip", name="IC"))
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    fig.update_xaxes(type="date")
    add_ramadan_zones(fig)
    return fig


def network_rules_chart(rules_df: pd.DataFrame, antecedent_col: str = "antecedent", consequent_col: str = "consequent", lift_col: str = "lift", title: str = "Réseau des règles", subtitle: str = ""):
    # Very compact circular layout for top rules.
    nodes = []
    links = []
    labels = []
    unique_items = []
    for _, row in rules_df.head(15).iterrows():
        a = str(row[antecedent_col])
        c = str(row[consequent_col])
        if a not in unique_items:
            unique_items.append(a)
        if c not in unique_items:
            unique_items.append(c)
        links.append((a, c, float(row[lift_col])))
    if not unique_items:
        return go.Figure()

    angles = np.linspace(0, 2 * np.pi, len(unique_items), endpoint=False)
    coords = {item: (np.cos(ang), np.sin(ang)) for item, ang in zip(unique_items, angles)}
    fig = go.Figure()
    for a, c, lift in links:
        xa, ya = coords[a]
        xb, yb = coords[c]
        fig.add_trace(go.Scatter(x=[xa, xb], y=[ya, yb], mode="lines", line=dict(color=COLORS["primary"], width=max(1.0, min(6.0, lift))), hoverinfo="skip", showlegend=False))
    xs, ys = zip(*[coords[item] for item in unique_items])
    fig.add_trace(go.Scatter(
        x=list(xs), y=list(ys), mode="markers+text", text=unique_items,
        textposition="top center", marker=dict(size=18, color=COLORS["primary_light"], line=dict(color=COLORS["primary"], width=1.5)),
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False)
    _add_plotly_title(fig, title, subtitle)
    apply_plotly_style(fig)
    return fig

