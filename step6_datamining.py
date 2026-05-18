#!/usr/bin/env python3
"""
step6_datamining.py

Add a lightweight data-mining stage to the pipeline:
- K-Means clustering of `CD_TYP_CONS` by monthly sales profile
- Apriori association rules on categorical attributes

Usage:
  python step6_datamining.py --input data_prepared_final.csv

Outputs (saved under `data/` and `figures/`):
  - data/datamining_clusters_by_type.csv
  - data/datamining_association_rules.csv
  - figures/clusters_profile.png

This script is defensive about column names and will try to auto-detect
common fields (CD_TYP_CONS, ARTES_VOL, YEAR/MONTH, REGION, ENERGIE, VP_VU).
"""

from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except Exception:
    apriori = None
    association_rules = None


def detect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data(path: Path):
    df = pd.read_csv(path)
    return df


def prepare_monthly_profile(df, type_col, vol_col):
    # Try to get or build a YYYY-MM period
    if 'DATE' in df.columns or 'Date' in df.columns:
        try:
            if 'DATE' in df.columns:
                df['__DATE'] = pd.to_datetime(df['DATE'])
            else:
                df['__DATE'] = pd.to_datetime(df['Date'])
            df['__PERIOD'] = df['__DATE'].dt.to_period('M').dt.to_timestamp()
        except Exception:
            df['__PERIOD'] = df.get('MONTH', df.get('PERIOD'))
    elif {'YEAR', 'MONTH'}.issubset(df.columns):
        df['__PERIOD'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2) + '-01')
    else:
        # Fall back to any column that looks like a period
        period_col = detect_column(df, ['PERIOD', 'MONTH_YEAR', 'MOIS', 'MONTH'])
        if period_col:
            df['__PERIOD'] = df[period_col]
        else:
            df['__PERIOD'] = 0

    pivot = (
        df.groupby([type_col, '__PERIOD'])[vol_col]
        .sum()
        .unstack(fill_value=0)
    )
    return pivot


def run_kmeans(pivot, k_min=2, k_max=6, random_state=42):
    X = pivot.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_samples = len(pivot)
    if n_samples == 0:
        raise ValueError('No samples available for clustering')
    if n_samples == 1:
        return {
            'model': None,
            'labels': np.array([0]),
            'scaler': scaler,
            'best_k': 1,
            'best_score': np.nan,
        }

    best_k = None
    best_score = -np.inf
    best_labels = None
    best_model = None

    # Silhouette is defined only when 2 <= k <= n_samples - 1.
    max_k_for_silhouette = min(k_max, n_samples - 1)
    for k in range(k_min, max_k_for_silhouette + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(Xs)

        try:
            score = silhouette_score(Xs, labels)
        except Exception:
            score = -np.inf

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = km

    # Fallback when silhouette cannot be computed (e.g., only 2 samples).
    if best_labels is None:
        fallback_k = min(max(k_min, 1), n_samples)
        km = KMeans(n_clusters=fallback_k, random_state=random_state, n_init=10)
        best_labels = km.fit_predict(Xs)
        best_model = km
        best_k = fallback_k
        best_score = np.nan

    return {
        'model': best_model,
        'labels': best_labels,
        'scaler': scaler,
        'best_k': best_k,
        'best_score': best_score,
    }


def plot_cluster_profiles(pivot, labels, out_path: Path):
    dfp = pivot.copy()
    dfp['cluster'] = labels
    
    # Aggregate to yearly volumes
    yearly_data = {}
    for col in dfp.columns:
        if col != 'cluster':
            try:
                # Try to extract year from the period column
                if isinstance(col, pd.Timestamp):
                    year = col.year
                elif hasattr(col, 'year'):
                    year = col.year
                else:
                    # Fallback: treat col as a string or numeric period
                    year_str = str(col).split('-')[0] if '-' in str(col) else str(col)[:4]
                    try:
                        year = int(year_str)
                    except:
                        year = col
                
                if year not in yearly_data:
                    yearly_data[year] = {}
                yearly_data[year][col] = dfp[col]
            except:
                pass
    
    # Reshape to yearly aggregated view
    yearly_pivot = pd.DataFrame()
    for year in sorted(yearly_data.keys()):
        yearly_pivot[str(year)] = dfp[[c for c in yearly_data[year].keys()]].sum(axis=1)
    
    if yearly_pivot.empty:
        yearly_pivot = dfp.iloc[:, dfp.columns != 'cluster'].copy()
    
    yearly_pivot['cluster'] = labels
    
    # Generate meaningful cluster names based on characteristics
    cluster_names = {}
    for cluster_id in sorted(yearly_pivot['cluster'].unique()):
        cluster_mask = yearly_pivot['cluster'] == cluster_id
        cluster_volumes = yearly_pivot.loc[cluster_mask, yearly_pivot.columns != 'cluster'].values.flatten()
        cluster_volumes = cluster_volumes[cluster_volumes > 0]  # Exclude zeros
        
        if len(cluster_volumes) > 0:
            avg_vol = cluster_volumes.mean()
            vol_std = cluster_volumes.std()
            vol_min = cluster_volumes.min()
            vol_max = cluster_volumes.max()
            trend = "↑" if vol_max > avg_vol * 1.1 else ("↓" if vol_min < avg_vol * 0.9 else "→")
            
            # Assign meaningful name based on volume level and trend
            if avg_vol > cluster_volumes.mean() * 1.2:
                level = "High"
            elif avg_vol < cluster_volumes.mean() * 0.8:
                level = "Low"
            else:
                level = "Medium"
            
            volatility = "Volatile" if vol_std > avg_vol * 0.3 else "Stable"
            
            cluster_names[cluster_id] = f"{level}-Volume {volatility} {trend}"
        else:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
    
    # Plot yearly aggregated data
    centers = yearly_pivot.groupby('cluster').mean().T
    
    plt.figure(figsize=(12, 5 + 0.3 * max(1, centers.shape[1])))
    for c in sorted(centers.columns):
        label_name = cluster_names.get(c, f'Cluster {c}')
        plt.plot(centers.index.astype(str), centers[c], marker='o', linewidth=2, markersize=8, label=label_name)
    
    plt.xticks(rotation=45)
    plt.title('Cluster profiles (yearly volume aggregated)')
    plt.xlabel('Year')
    plt.ylabel('Total yearly volume')
    plt.grid(True, alpha=0.3)
    if centers.shape[1] > 0:
        plt.legend(loc='best')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _prepare_apriori_features(df, type_col, vol_col):
    work = df.copy()

    # Add a sales-level bucket to create more actionable rules.
    if vol_col in work.columns and pd.api.types.is_numeric_dtype(work[vol_col]):
        series = work[vol_col].dropna()
        if series.nunique() >= 4:
            try:
                work['VENTES_NIVEAU'] = pd.qcut(
                    work[vol_col],
                    q=4,
                    labels=['faible', 'moyen', 'eleve', 'tres_eleve'],
                    duplicates='drop',
                )
            except Exception:
                pass

    # Add a compact seasonal label instead of raw month/quarter combinations.
    if 'MOIS' in work.columns:
        month_to_season = {
            12: 'hiver', 1: 'hiver', 2: 'hiver',
            3: 'printemps', 4: 'printemps', 5: 'printemps',
            6: 'ete', 7: 'ete', 8: 'ete',
            9: 'automne', 10: 'automne', 11: 'automne',
        }
        try:
            work['SAISON'] = work['MOIS'].map(month_to_season)
        except Exception:
            pass

    # Prefer non-deterministic features to avoid tautological rules.
    candidate_cols = [
        type_col,
        'SAISON',
        'EST_RAMADAN',
        'EST_WEEKEND',
        'EST_VACANCES_SCOLAIRES',
        'EST_FIN_MOIS',
        'EST_DEBUT_MOIS',
        'VENTES_NIVEAU',
    ]

    present = [c for c in candidate_cols if c in work.columns]

    # Keep only columns with meaningful variation.
    selected = []
    for c in present:
        nunique = work[c].nunique(dropna=True)
        if 2 <= nunique <= 20:
            selected.append(c)

    if len(selected) < 2:
        print(f'⚠️  Apriori will be skipped: only {len(selected)} categorical feature(s) with sufficient variation found.')
        if selected:
            print(f'    Found: {selected}')
        print(f'    Need: at least 2 features for meaningful association rules')
        return work, []
    return work, selected


def run_apriori(df, categorical_columns, min_support=0.03, min_threshold=0.35):
    if apriori is None:
        raise RuntimeError('mlxtend is required for apriori; please install mlxtend')

    # Build transactions: one transaction per row, items are col=val strings
    trans = df[categorical_columns].astype(str).fillna('NA')
    prefixed = pd.DataFrame({c: c + '=' + trans[c] for c in categorical_columns})
    ohe = pd.get_dummies(prefixed.astype(str), prefix='', prefix_sep='').astype(bool)

    frequent = apriori(ohe, min_support=min_support, use_colnames=True, max_len=3)
    rules = association_rules(frequent, metric='confidence', min_threshold=min_threshold)

    # If confidence is too strict for this dataset, relax it once.
    if rules.empty and min_threshold > 0.2:
        rules = association_rules(frequent, metric='confidence', min_threshold=0.2)

    if not rules.empty:
        # Keep readable rules with limited complexity.
        rules = rules[
            (rules['antecedents'].apply(len) <= 2)
            & (rules['consequents'].apply(len) == 1)
            & (rules['lift'] > 1.01)
        ]
        rules = rules.sort_values(['lift', 'confidence', 'support'], ascending=[False, False, False])

    return frequent, rules


def _itemset_to_text(itemset):
    if isinstance(itemset, (set, frozenset)):
        return ' + '.join(sorted(str(x) for x in itemset))
    return str(itemset)


def export_top_rules(rules, out_dir: Path, top_n=15):
    if rules is None or rules.empty:
        return None, None

    cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    work = rules[cols].copy()
    work['antecedent'] = work['antecedents'].apply(_itemset_to_text)
    work['consequent'] = work['consequents'].apply(_itemset_to_text)

    # Keep one strongest rule per exact antecedent -> consequent pair.
    work = work.sort_values(['lift', 'confidence', 'support'], ascending=[False, False, False])
    work = work.drop_duplicates(subset=['antecedent', 'consequent'], keep='first')

    top = work[['antecedent', 'consequent', 'support', 'confidence', 'lift']].head(top_n).copy()
    top['support'] = top['support'].round(4)
    top['confidence'] = top['confidence'].round(4)
    top['lift'] = top['lift'].round(4)

    csv_path = out_dir / 'datamining_association_rules_top15.csv'
    md_path = out_dir / 'datamining_association_rules_top15.md'
    top.to_csv(csv_path, index=False)

    header = '| antecedent | consequent | support | confidence | lift |\n'
    sep = '|---|---|---:|---:|---:|\n'
    lines = [header, sep]
    for _, row in top.iterrows():
        line = (
            f"| {row['antecedent']} | {row['consequent']} | "
            f"{row['support']:.4f} | {row['confidence']:.4f} | {row['lift']:.4f} |\n"
        )
        lines.append(line)
    md_path.write_text(''.join(lines), encoding='utf-8')

    return csv_path, md_path


def main(argv):
    p = argparse.ArgumentParser(description='Data mining: clustering + apriori')
    p.add_argument('--input', default='data_prepared_final.csv')
    p.add_argument('--out-dir', default='data')
    p.add_argument('--fig-dir', default='figures')
    p.add_argument('--kmin', type=int, default=2)
    p.add_argument('--kmax', type=int, default=6)
    args = p.parse_args(argv)

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    if not inp.exists():
        print(f'Input file not found: {inp}', file=sys.stderr)
        sys.exit(2)

    df = load_data(inp)

    # Detect columns (include French variants commonly present in datasets)
    type_col = detect_column(df, ['CD_TYP_CONS', 'TYPE_MARCHE', 'TYPE', 'TYPE_CONS', 'CD_TYPE'])
    vol_col = detect_column(df, ['ARTES_VOL', 'VENTES', 'VOL', 'VOLUME', 'volume', 'qty'])

    if type_col is None or vol_col is None:
        print('Could not detect necessary columns (type, volume).', file=sys.stderr)
        print('Columns available:', df.columns.tolist(), file=sys.stderr)
        sys.exit(3)

    pivot = prepare_monthly_profile(df, type_col, vol_col)

    # KMeans
    km_res = run_kmeans(pivot, k_min=args.kmin, k_max=args.kmax)
    labels = km_res['labels']
    best_k = km_res['best_k']
    best_score = km_res['best_score']

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    clusters_df = pd.DataFrame({
        type_col: pivot.index,
        'cluster': labels
    })
    clusters_path = out_dir / 'datamining_clusters_by_type.csv'
    clusters_df.to_csv(clusters_path, index=False)
    if pd.isna(best_score):
        print(f'Wrote clusters to {clusters_path} (k={best_k}, silhouette=NA)')
    else:
        print(f'Wrote clusters to {clusters_path} (k={best_k}, silhouette={best_score:.3f})')

    plot_cluster_profiles(pivot, labels, fig_dir / 'clusters_profile.png')
    print(f'Wrote cluster profile figure to {fig_dir / "clusters_profile.png"}')

    # Apriori: choose categorical columns with enough variation.
    apriori_df, cat_cols = _prepare_apriori_features(df, type_col, vol_col)

    if len(cat_cols) >= 2 and apriori is not None:
        frequent, rules = run_apriori(apriori_df, cat_cols)
        freq_path = out_dir / 'datamining_frequent_itemsets.csv'
        rules_path = out_dir / 'datamining_association_rules.csv'
        frequent.to_csv(freq_path, index=False)
        rules.to_csv(rules_path, index=False)
        top_csv, top_md = export_top_rules(rules, out_dir, top_n=15)
        print(
            f'Wrote frequent itemsets to {freq_path} and rules to {rules_path} '
            f'(columns used: {", ".join(cat_cols)})'
        )
        if top_csv and top_md:
            print(f'Wrote top clean rules to {top_csv} and {top_md}')
    else:
        if apriori is None:
            print('⚠️  Apriori skipped: mlxtend not installed. Run: pip install mlxtend')
        else:
            print(f'⚠️  Apriori skipped: insufficient categorical features (found {len(cat_cols)} with variation, need ≥2).')
            if cat_cols:
                print(f'    Available features: {cat_cols}')


if __name__ == '__main__':
    main(sys.argv[1:])
