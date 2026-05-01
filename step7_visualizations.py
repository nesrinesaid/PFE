"""
STEP 7 - VISUALIZATION OF FORECASTING RESULTS

This script converts step6 CSV outputs into publication-ready charts.
Generates beautiful visualizations for thesis, presentations, and reports.

Usage:
    python step7_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Chart dimensions and styling
CHART_WIDTH = 12
CHART_HEIGHT = 6
CHART_HEIGHT_TALL = 12
MARKER_SIZE = 8
MARKER_SIZE_SMALL = 7
LINE_WIDTH = 2.5
LINE_WIDTH_THIN = 2
BAR_EDGE_WIDTH = 1.5
DPI = 300
ALPHA_BAR = 0.8
ALPHA_FILL = 0.3
ALPHA_GRID = 0.3

# Set professional style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Color scheme (professional)
COLORS = {
    'vp': '#1f77b4',          # Steel blue
    'vu': '#ff7f0e',          # Orange
    'total': '#2ca02c',       # Green
    'artes': '#d62728',       # Red
    'baseline': '#1f77b4',    # Blue
    'optimiste': '#2ca02c',   # Green
    'prudent': '#d62728',     # Red
}

# ============================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================

def main():
    print("="*70)
    print("STEP 7 - CREATING VISUALIZATIONS FROM FORECAST DATA")
    print("="*70)
    print()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # ========== Load forecast data ==========
    print("Loading forecast data...")
    forecast_file = os.path.join(project_root, 'step6_forecast_s1_2026.csv')

    # fallback: use baseline scenario if combined file not present
    if not os.path.exists(forecast_file):
        fallback = os.path.join(project_root, 'step6_forecast_s1_2026_baseline.csv')
        if os.path.exists(fallback):
            forecast_file = fallback
        else:
            print(f"❌ ERROR: No forecast file found (neither step6_forecast_s1_2026.csv nor baseline).")
            print("   Run step6_modeling.py first.")
            return

    forecast = pd.read_csv(forecast_file)
    # Validate forecast data
    if forecast.empty:
        print("❌ ERROR: Forecast data is empty.")
        return
    if forecast['Date'].isna().any():
        print("⚠️  WARNING: Some dates are NaN. Cleaning...")
        forecast = forecast.dropna(subset=['Date'])
    forecast['Date'] = pd.to_datetime(forecast['Date'])
    if len(forecast) == 0:
        print("❌ ERROR: No valid forecast rows after date validation.")
        return
    print(f"✅ Loaded forecast data: {len(forecast)} rows")
    print()

    # ========== CHART 1: Total Market Monthly Forecast ==========
    print("Creating CHART 1: Total Market Forecast...")
    create_chart_total_market(forecast, project_root)
    
    # ========== CHART 2: VP vs VU Breakdown ==========
    print("Creating CHART 2: VP vs VU Breakdown...")
    create_chart_vp_vu_breakdown(forecast, project_root)
    
    # ========== CHART 3: ARTES Brand Volume ==========
    print("Creating CHART 3: ARTES Brand Volume...")
    create_chart_artes_volume(forecast, project_root)
    
    # ========== CHART 4: Market Share Evolution ==========
    print("Creating CHART 4: Market Share Evolution...")
    create_chart_market_share(forecast, project_root)
    
    # ========== CHART 5: Scenario Comparison (if available) ==========
    print("Creating CHART 5: Scenario Comparison...")
    create_chart_scenarios(project_root)
    
    # ========== CHART 6: Complete Dashboard ==========
    print("Creating CHART 6: Complete Summary Dashboard...")
    create_dashboard_4panel(forecast, project_root)
    
    print()
    print("="*70)
    print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY")
    print("="*70)
    print()
    print("Generated files:")
    print("  1. 13_Forecast_Total_Market.png")
    print("  2. 14_Forecast_VP_vs_VU.png")
    print("  3. 15_Forecast_ARTES_Volume.png")
    print("  4. 16_Forecast_Market_Share.png")
    print("  5. 17_Forecast_Scenarios_Comparison.png")
    print("  6. 18_Forecast_Complete_Dashboard.png")
    print()
    print("Ready for thesis, presentations, and reports! 📊")
    print()


# ============================================================
# INDIVIDUAL CHART FUNCTIONS
# ============================================================

def create_chart_total_market(forecast, project_root):
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    
    # Create bar chart
    bars = ax.bar(
        range(len(forecast)),
        forecast.get('PREV_TOTAL_MARCHE', forecast.get('TOTAL_MARCHE', np.nan)),
        color=COLORS['total'],
        edgecolor='black',
        linewidth=BAR_EDGE_WIDTH,
        alpha=ALPHA_BAR
    )
    
    # Add value labels on bars
    values = forecast.get('PREV_TOTAL_MARCHE', forecast.get('TOTAL_MARCHE', np.nan))
    for i, value in enumerate(values):
        try:
            ax.text(i, value + max(values)*0.03, f"{int(value):,}", ha='center', va='bottom', fontweight='bold', fontsize=11)
        except Exception:
            pass
    
    ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de Véhicules Vendus', fontsize=12, fontweight='bold')
    ax.set_title('Prévision du Marché Total - S1 2026', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(forecast)))
    ax.set_xticklabels(forecast['Date'].dt.strftime('%b %Y'), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    avg_value = np.nanmean(values)
    if not np.isnan(avg_value):
        ax.axhline(y=avg_value, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Moyenne: {int(avg_value):,}')
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(project_root, '13_Forecast_Total_Market.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 13_Forecast_Total_Market.png")


def create_chart_vp_vu_breakdown(forecast, project_root):
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    x = np.arange(len(forecast))
    width = 0.35
    vp = forecast.get('PREV_VP', forecast.get('VP', np.zeros(len(forecast))))
    vu = forecast.get('PREV_VU', forecast.get('VU', np.zeros(len(forecast))))
    bars1 = ax.bar(x - width/2, vp, width, label='VP (Particuliers)', color=COLORS['vp'], edgecolor='black', linewidth=BAR_EDGE_WIDTH, alpha=ALPHA_BAR)
    bars2 = ax.bar(x + width/2, vu, width, label='VU (Utilitaires)', color=COLORS['vu'], edgecolor='black', linewidth=BAR_EDGE_WIDTH, alpha=ALPHA_BAR)
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de Véhicules', fontsize=12, fontweight='bold')
    ax.set_title('Décomposition VP / VU - S1 2026', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(forecast['Date'].dt.strftime('%b %Y'), rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    output_path = os.path.join(project_root, '14_Forecast_VP_vs_VU.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 14_Forecast_VP_vs_VU.png")


def create_chart_artes_volume(forecast, project_root):
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    artes = forecast.get('PREV_VOL_ARTES', forecast.get('VOL_ARTES', np.zeros(len(forecast))))
    ax.plot(forecast['Date'], artes, marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, color=COLORS['artes'], label='Volume ARTES')
    ax.fill_between(forecast['Date'], artes, alpha=ALPHA_FILL, color=COLORS['artes'])
    for idx, row in forecast.iterrows():
        try:
            ax.text(row['Date'], artes.iloc[idx] + max(artes)*0.03, f"{int(artes.iloc[idx]):,}", ha='center', va='bottom', fontweight='bold', fontsize=10)
        except Exception:
            pass
    ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume de Ventes', fontsize=12, fontweight='bold')
    ax.set_title('Volume ARTES (Renault + Dacia + Nissan) - S1 2026', fontsize=14, fontweight='bold', pad=20)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    output_path = os.path.join(project_root, '15_Forecast_ARTES_Volume.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 15_Forecast_ARTES_Volume.png")


def create_chart_market_share(forecast, project_root):
    """
    Chart 4: Market share evolution for VP, VU, and ARTES
    Shows all three key market segments over time
    """
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    
    # Get market shares
    part_vp = forecast.get('PART_VP', np.zeros(len(forecast)))
    part_vu = 1 - part_vp  # VU share is complement of VP share
    part_artes = forecast.get('PREV_PART_ARTES', forecast.get('PART_ARTES', np.zeros(len(forecast))))
    
    # Plot all three shares
    ax.plot(forecast['Date'], part_vp * 100, 
            marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
            color=COLORS['vp'], label='Part VP (%)', alpha=0.85)
    
    ax.plot(forecast['Date'], part_vu * 100, 
            marker='s', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
            color=COLORS['vu'], label='Part VU (%)', alpha=0.85)
    
    ax.plot(forecast['Date'], part_artes * 100, 
            marker='^', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
            color=COLORS['artes'], label='Part ARTES (%)', alpha=0.85)
    
    # Formatting
    ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
    ax.set_ylabel('Part de Marché (%)', fontsize=12, fontweight='bold')
    ax.set_title('Évolution des Parts de Marché - S1 2026 (VP, VU, ARTES)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=ALPHA_GRID, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = os.path.join(project_root, '16_Forecast_Market_Share.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 16_Forecast_Market_Share.png")


def create_chart_scenarios(project_root):
    """
    Chart 5: Scenario comparison - baseline vs optimiste vs prudent
    Shows best case, normal case, and worst case forecasts
    """
    baseline_file = os.path.join(project_root, 'step6_forecast_s1_2026_baseline.csv')
    optimiste_file = os.path.join(project_root, 'step6_forecast_s1_2026_optimiste.csv')
    prudent_file = os.path.join(project_root, 'step6_forecast_s1_2026_prudent.csv')
    
    # Check if baseline exists (required)
    if not os.path.exists(baseline_file):
        print(f"  ⚠️  Baseline scenario file not found.")
        print(f"     Run: python add_macro_scenarios.py && python step6_modeling.py")
        return
    
    # Load baseline (required)
    print(f"  ✅ Loading baseline scenario...")
    baseline = pd.read_csv(baseline_file)
    baseline['Date'] = pd.to_datetime(baseline['Date'])
    
    # Load optimiste (optional)
    optimiste = None
    if os.path.exists(optimiste_file):
        optimiste = pd.read_csv(optimiste_file)
        optimiste['Date'] = pd.to_datetime(optimiste['Date'])
        print(f"  ✅ Loaded optimiste scenario")
    else:
        print(f"  ⚠️  Optimiste file not found")
    
    # Load prudent (optional)
    prudent = None
    if os.path.exists(prudent_file):
        prudent = pd.read_csv(prudent_file)
        prudent['Date'] = pd.to_datetime(prudent['Date'])
        print(f"  ✅ Loaded prudent scenario")
    else:
        print(f"  ⚠️  Prudent file not found")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    
    # Plot baseline (always)
    ax.plot(baseline['Date'], baseline['PREV_TOTAL_MARCHE'], 
            marker='o', linewidth=3, markersize=9, 
            color=COLORS['baseline'], label='Baseline (Normal)', alpha=0.9, zorder=3)
    
    # Plot optimiste (if available)
    if optimiste is not None:
        ax.plot(optimiste['Date'], optimiste['PREV_TOTAL_MARCHE'], 
                marker='s', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
                color=COLORS['optimiste'], label='Optimiste (Strong Growth)', alpha=0.85, zorder=2)
    
    # Plot prudent (if available)
    if prudent is not None:
        ax.plot(prudent['Date'], prudent['PREV_TOTAL_MARCHE'], 
                marker='^', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
                color=COLORS['prudent'], label='Prudent (Weak Growth)', alpha=0.85, zorder=2)
    
    # Fill between optimiste and prudent if both available
    if (optimiste is not None) and (prudent is not None):
        ax.fill_between(baseline['Date'], 
                        prudent['PREV_TOTAL_MARCHE'], 
                        optimiste['PREV_TOTAL_MARCHE'],
                        alpha=ALPHA_FILL, color='gray', label='Plage de prévision', zorder=1)
    
    # Formatting
    ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Marché (Ventes)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Scénarios Macroéconomiques - S1 2026', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=ALPHA_GRID, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = os.path.join(project_root, '17_Forecast_Scenarios_Comparison.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 17_Forecast_Scenarios_Comparison.png")


def create_dashboard_4panel(forecast, project_root):
    fig, axes = plt.subplots(2, 2, figsize=(16, CHART_HEIGHT_TALL))
    fig.suptitle('PRÉVISIONS ARTES - S1 2026 - TABLEAU DE BORD COMPLET', fontsize=16, fontweight='bold', y=0.995)
    # Panel 1
    ax = axes[0, 0]
    bars = ax.bar(range(len(forecast)), forecast.get('PREV_TOTAL_MARCHE', forecast.get('TOTAL_MARCHE', np.zeros(len(forecast)))), color=COLORS['total'], edgecolor='black', linewidth=BAR_EDGE_WIDTH, alpha=ALPHA_BAR)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(i, height + max(forecast.get('PREV_TOTAL_MARCHE', forecast.get('TOTAL_MARCHE', np.zeros(len(forecast))))) * 0.02, f"{int(height):,}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.set_title('Marché Total Mensuel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ventes', fontsize=11)
    ax.set_xticks(range(len(forecast)))
    ax.set_xticklabels(forecast['Date'].dt.strftime('%b'), rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    # Panel 2
    ax = axes[0, 1]
    x = np.arange(len(forecast))
    width = 0.35
    ax.bar(x - width/2, forecast.get('PREV_VP', np.zeros(len(forecast))), width, label='VP', color=COLORS['vp'], alpha=ALPHA_BAR, edgecolor='black', linewidth=BAR_EDGE_WIDTH)
    ax.bar(x + width/2, forecast.get('PREV_VU', np.zeros(len(forecast))), width, label='VU', color=COLORS['vu'], alpha=ALPHA_BAR, edgecolor='black', linewidth=BAR_EDGE_WIDTH)
    ax.set_title('Décomposition VP / VU', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ventes', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(forecast['Date'].dt.strftime('%b'), rotation=45)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    # Panel 3
    ax = axes[1, 0]
    ax.plot(forecast['Date'], forecast.get('PREV_VOL_ARTES', np.zeros(len(forecast))), marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, color=COLORS['artes'], label='ARTES')
    ax.fill_between(forecast['Date'], forecast.get('PREV_VOL_ARTES', np.zeros(len(forecast))), alpha=ALPHA_FILL, color=COLORS['artes'])
    ax.set_title('Volume ARTES (Renault+Dacia+Nissan)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ventes', fontsize=11)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    # Panel 4
    ax = axes[1, 1]
    ax.plot(forecast['Date'], forecast.get('PART_VP', np.zeros(len(forecast))) * 100, marker='s', linewidth=LINE_WIDTH_THIN, markersize=MARKER_SIZE_SMALL, color=COLORS['vp'], label='Part VP (%)', alpha=0.8)
    ax.plot(forecast['Date'], forecast.get('PREV_PART_ARTES', np.zeros(len(forecast))) * 100, marker='^', linewidth=LINE_WIDTH_THIN, markersize=MARKER_SIZE_SMALL, color=COLORS['artes'], label='Part ARTES (%)', alpha=0.8)
    ax.set_title('Évolution des Parts de Marché', fontsize=12, fontweight='bold')
    ax.set_ylabel('Part (%)', fontsize=11)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.legend(fontsize=10)
    ax.grid(alpha=ALPHA_GRID)
    ax.set_axisbelow(True)
    ax.set_ylim([0, 100])
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_path = os.path.join(project_root, '18_Forecast_Complete_Dashboard.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: 18_Forecast_Complete_Dashboard.png")


if __name__ == '__main__':
    main()
