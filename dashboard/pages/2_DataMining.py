from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from components import auth, charts, data_loader, header
from components.export import render_export_panel

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except Exception:
    HAS_MLXTEND = False

st.set_page_config(page_title="Data Mining — ARTES", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

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


@st.cache_data(show_spinner=False)
def _load_source() -> pd.DataFrame:
    try:
        df = data_loader.load_cleaned_enriched().copy()
        df["Date"] = pd.to_datetime(df["DATV"], errors="coerce")
        if "VENTES" not in df.columns:
            if "IM_RI" in df.columns:
                df["VENTES"] = pd.to_numeric(df["IM_RI"], errors="coerce").fillna(0.0)
            else:
                df["VENTES"] = 1
        
        # Filter for new vehicles (IM_RI=10) and VP/VU market types
        if "IM_RI" in df.columns:
            df = df[pd.to_numeric(df["IM_RI"], errors="coerce") == 10].copy()
        
        if "TYPE_MARCHE" in df.columns:
            df = df[df["TYPE_MARCHE"].astype(str).str.upper().isin(["VP", "VU"])].copy()
            
    except Exception:
        df = data_loader.load_monthly_prepared().copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "VENTES" not in df.columns:
            df["VENTES"] = 1
        else:
            df["VENTES"] = pd.to_numeric(df["VENTES"], errors="coerce").fillna(0.0)

    for col in ["MARQUE", "CONTINENT", "ENERGIE", "SEGMENT", "TYPE_MARCHE", "MODELE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan, "0": np.nan})

    df["ANNEE"] = df["Date"].dt.year
    df["MOIS"] = df["Date"].dt.month
    return df.dropna(subset=["Date"]).copy()


def _brand_features(df: pd.DataFrame) -> pd.DataFrame:
    brand_col = None
    for candidate in ["MARQUE", "MODELE", "TYPE_MARCHE"]:
        if candidate in df.columns and df[candidate].notna().any():
            brand_col = candidate
            break

    if brand_col is None:
        df = df.copy()
        df["MARQUE"] = "Groupe_Observations"
        brand_col = "MARQUE"

    monthly = df.groupby([brand_col, "MOIS"], as_index=False)["VENTES"].sum().pivot(index=brand_col, columns="MOIS", values="VENTES").fillna(0.0)
    monthly.columns = [f"MOIS_{int(c)}" for c in monthly.columns]
    stats = pd.DataFrame(index=monthly.index)
    stats["TOTAL"] = monthly.sum(axis=1)
    stats["MOYENNE"] = monthly.mean(axis=1)
    stats["STD"] = monthly.std(axis=1)
    stats["CV"] = np.where(stats["MOYENNE"] > 0, stats["STD"] / stats["MOYENNE"], 0.0)
    stats["PIC_MOIS"] = monthly.idxmax(axis=1).str.replace("MOIS_", "").astype(int)
    stats["PIC_SHARE"] = np.where(stats["TOTAL"] > 0, monthly.max(axis=1) / stats["TOTAL"], 0.0)
    x = np.arange(1, 13)
    slopes = []
    for _, row in monthly.iterrows():
        coeffs = np.polyfit(x, row.values, 1)
        slopes.append(coeffs[0])
    stats["PENTE"] = slopes
    if "CONTINENT" in df.columns:
        continent_mode = df.groupby(brand_col)["CONTINENT"].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Inconnu")
        stats["CONTINENT"] = continent_mode
    else:
        stats["CONTINENT"] = "Inconnu"
    if "TYPE_MARCHE" in df.columns:
        type_mode = df.groupby(brand_col)["TYPE_MARCHE"].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "VP")
        stats["TYPE_DOMINANT"] = type_mode
    else:
        stats["TYPE_DOMINANT"] = "VP"
    if "SEGMENT" in df.columns:
        segment_mode = df.groupby(brand_col)["SEGMENT"].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Inconnu")
        stats["SEGMENT"] = segment_mode
    else:
        stats["SEGMENT"] = "Inconnu"
    if "ENERGIE" in df.columns:
        energy_mode = df.groupby(brand_col)["ENERGIE"].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Inconnu")
        stats["ENERGIE"] = energy_mode
    else:
        stats["ENERGIE"] = "Inconnu"
    result = stats.join(monthly)
    result = result.reset_index().rename(columns={"index": "MARQUE", brand_col: "MARQUE"})
    return result


df = _load_source()
st.title("Data Mining — Véhicules neufs VP/VU")
st.caption("Question décisionnelle : Quels patterns structurent le marché des véhicules neufs (IM_RI=10) sur les segments VP/VU ?")

try:
    cluster_df = _brand_features(df)
except Exception as e:
    st.error(f"❌ Erreur lors du traitement des données brutes: {str(e)}")
    st.stop()

# Initialize all variables that might be used later
summary_df = pd.DataFrame()
fig_elbow = None
fig_pca = None
fig_radar = None
work = pd.DataFrame()
elbow_df = pd.DataFrame()
K = 4
selected_features = []

st.sidebar.header("Clustering K-Means")
K = st.sidebar.slider("Nombre de clusters (K)", 2, 8, 4)
all_features = ["TOTAL", "MOYENNE", "STD", "CV", "PIC_MOIS", "PIC_SHARE", "PENTE"]
selected_features = st.sidebar.multiselect("Features", all_features, default=["TOTAL", "MOYENNE", "STD", "CV", "PIC_MOIS", "PENTE"])

if not selected_features:
    selected_features = ["TOTAL", "MOYENNE", "STD", "CV", "PIC_MOIS", "PENTE"]

work = cluster_df[["MARQUE"] + selected_features + ["CONTINENT", "TYPE_DOMINANT", "SEGMENT", "ENERGIE"]].copy()
X = StandardScaler().fit_transform(work[selected_features].fillna(0.0))

max_k = min(8, len(work) - 1) if len(work) > 2 else 2
inertias = []
for k in range(2, max_k + 1):
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_tmp.fit(X)
    inertias.append({"K": k, "Inertia": km_tmp.inertia_})
elbow_df = pd.DataFrame(inertias)

km = KMeans(n_clusters=min(K, len(work)), random_state=42, n_init=10)
work["cluster"] = km.fit_predict(X)
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X)
work["PC1"] = pcs[:, 0]
work["PC2"] = pcs[:, 1]

fig_elbow = go.Figure(go.Scatter(x=elbow_df["K"], y=elbow_df["Inertia"], mode="lines+markers", line=dict(color=charts.COLORS["primary"], width=2.5), marker=dict(size=7)))
fig_elbow.update_layout(title="Justification du K optimal", xaxis_title="K", yaxis_title="Inertie")
charts.apply_plotly_style(fig_elbow)

fig_pca = charts.scatter_pca(work, x_col="PC1", y_col="PC2", cluster_col="cluster", label_col="MARQUE", title="Clusters PCA 2D", subtitle="Projection des clusters")

st.subheader("6.1 — Clustering K-Means")
c1, c2 = st.columns([1.1, 1])
with c1:
    st.markdown("#### Courbe du coude")
    st.plotly_chart(fig_elbow, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
with c2:
    st.markdown("#### Scatter PCA 2D")
    st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})

st.markdown("#### Profil radar par cluster")
radar_input = work[["cluster"] + selected_features].copy()
fig_radar = charts.radar_profile(radar_input, "cluster", selected_features, title="Profil radar des clusters", subtitle="Moyennes normalisées par cluster")
st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})

summary_rows = []
for cluster_id, sub in work.groupby("cluster"):
    top5 = sub.sort_values("TOTAL", ascending=False).head(5)["MARQUE"].tolist()
    month_dominant = int(sub["PIC_MOIS"].mode().iloc[0]) if not sub["PIC_MOIS"].mode().empty else 0
    type_dom = sub["TYPE_DOMINANT"].mode().iloc[0] if not sub["TYPE_DOMINANT"].mode().empty else "VP"
    summary_rows.append({
        "Cluster": f"Cluster {cluster_id}",
        "Top 5 marques": ", ".join(top5),
        "Mois dominant": month_dominant,
        "Type dominant": type_dom,
    })
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, height=220)

st.markdown("#### Interprétation automatique")
for cluster_id, sub in work.groupby("cluster"):
    brands = ", ".join(sub.sort_values("TOTAL", ascending=False).head(2)["MARQUE"].tolist())
    cont = sub["CONTINENT"].mode().iloc[0] if not sub["CONTINENT"].mode().empty else "mixte"
    peak = int(sub["PIC_MOIS"].mode().iloc[0]) if not sub["PIC_MOIS"].mode().empty else 0
    type_dom = sub["TYPE_DOMINANT"].mode().iloc[0] if not sub["TYPE_DOMINANT"].mode().empty else "VP"
    st.info(f"Cluster {cluster_id} : véhicules {type_dom} à dominante {cont.lower()}, pic en mois {peak}, conduit par {brands}.")

st.markdown("---")
st.subheader("6.2 — Règles d'Association")
st.sidebar.header("Association Rules")
support = st.sidebar.slider("Support minimum", 0.01, 0.5, 0.05)
confidence = st.sidebar.slider("Confiance minimum", 0.1, 1.0, 0.3)
item_choices = st.sidebar.multiselect("Items", ["Marché", "Région", "Énergie", "Mois", "Marque"], default=["Marché", "Région", "Énergie", "Mois"])

# Initialize variables that might be used later
display_cols = ["Antécédent → Conséquent", "support", "confidence", "lift"]
rules = pd.DataFrame()
fig_rules = None

if not HAS_MLXTEND:
    st.error("mlxtend n'est pas disponible dans l'environnement courant.")
else:
    tx = pd.DataFrame(index=df.index)
    if "Marché" in item_choices and "TYPE_MARCHE" in df.columns:
        tx["Marché"] = "Marché=" + df["TYPE_MARCHE"].astype(str)
    if "Région" in item_choices and "CONTINENT" in df.columns:
        tx["Région"] = "Région=" + df["CONTINENT"].astype(str)
    if "Énergie" in item_choices and "ENERGIE" in df.columns:
        tx["Énergie"] = "Énergie=" + df["ENERGIE"].astype(str)
    if "Mois" in item_choices and "MOIS" in df.columns:
        tx["Mois"] = "Mois=" + df["MOIS"].astype(int).astype(str)
    if "Marque" in item_choices and "MARQUE" in df.columns:
        tx["Marque"] = "Marque=" + df["MARQUE"].astype(str)

    tx = tx.dropna(axis=1, how="all")
    if tx.shape[1] < 2:
        st.warning("Sélectionnez au moins deux types d'items pour générer des règles.")
    else:
        basket = pd.get_dummies(tx.astype(str), prefix="", prefix_sep="").astype(bool)
        freq = apriori(basket, min_support=support, use_colnames=True, max_len=3)
        rules = association_rules(freq, metric="confidence", min_threshold=confidence) if not freq.empty else pd.DataFrame()
        if not rules.empty:
            rules = rules[(rules["lift"] > 1.0)].sort_values(["lift", "confidence", "support"], ascending=False)
            rules["antecedent"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(list(x))))
            rules["consequent"] = rules["consequents"].apply(lambda x: " + ".join(sorted(list(x))))
            rules["Antécédent → Conséquent"] = rules["antecedent"] + " → " + rules["consequent"]
            display_cols = ["Antécédent → Conséquent", "support", "confidence", "lift"]
            styled = rules[display_cols].head(10).style.apply(lambda row: ["background-color: #EDFAF4" if row["lift"] > 2 else "" for _ in row], axis=1)
            st.dataframe(styled, use_container_width=True, height=360)
            fig_rules = charts.network_rules_chart(rules, antecedent_col="antecedent", consequent_col="consequent", lift_col="lift", title="Graphique réseau des règles", subtitle="Épaisseur proportionnelle au lift")
            st.plotly_chart(fig_rules, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": "artes_chart"}})
            artis_mask = rules["Antécédent → Conséquent"].str.contains("Marque=RENAULT|Marque=DACIA", case=False, regex=True)
            artis_rules = rules[artis_mask].head(3)
            if not artis_rules.empty:
                st.success("Insight ARTES : règles impliquant Renault/Dacia détectées.")
                st.dataframe(artis_rules[display_cols], use_container_width=True, height=180)
            else:
                st.info("Insight ARTES : aucune règle forte Renault/Dacia détectée dans ce filtrage.")
        else:
            st.warning("Aucune règle significative trouvée avec les paramètres actuels.")

render_export_panel(
    title="Data Mining — ARTES",
    prefix="datamining",
    dataframes={
        "clusters": work[["MARQUE", "cluster", "CONTINENT", "TYPE_DOMINANT", "SEGMENT", "ENERGIE"] + selected_features],
        "coude": elbow_df,
        "summary_clusters": summary_df,
        "rules": rules[display_cols].head(10) if HAS_MLXTEND and 'rules' in locals() and not rules.empty else pd.DataFrame(),
    },
    figures={
        "elbow": fig_elbow,
        "pca_clusters": fig_pca,
        "radar_clusters": fig_radar,
        "network_rules": fig_rules if HAS_MLXTEND and 'fig_rules' in locals() else None,
    },
    summary_lines=[
        f"Clusters K-Means: {K}",
        f"Features utilisées: {', '.join(selected_features)}",
        f"Support min: {support:.2f} | Confiance min: {confidence:.2f}",
    ],
)

auth.footer()
