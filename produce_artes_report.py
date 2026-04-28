import os
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.abspath(__file__))

scenario_files = [
    ("baseline", os.path.join(project_root, "step6_forecast_s1_2026_baseline.csv")),
    ("optimiste", os.path.join(project_root, "step6_forecast_s1_2026_optimiste.csv")),
    ("prudent", os.path.join(project_root, "step6_forecast_s1_2026_prudent.csv")),
]

dfs = {}
for name, path in scenario_files:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"]) if os.path.exists(path) else pd.DataFrame()
        if "PREV_TOTAL_MARCHE" in df.columns:
            dfs[name] = df[[("Date"), ("PREV_TOTAL_MARCHE")]].copy()
        else:
            # fallback if only one column present
            cols = [c for c in df.columns if c.lower().startswith("prev") or c.lower().startswith("total")]
            if cols:
                dfs[name] = df[["Date", cols[0]]].rename(columns={cols[0]: "PREV_TOTAL_MARCHE"}).copy()

if not dfs:
    print("No scenario forecast CSVs found. Run step6_modeling.py first.")
    raise SystemExit(1)

all_dates = sorted({d for df in dfs.values() for d in list(df["Date"])})
report_rows = []
monthly_table = pd.DataFrame({"Date": pd.to_datetime(all_dates)})
for name, df in dfs.items():
    df2 = df.set_index("Date").reindex(monthly_table["Date"]).ffill().fillna(0.0)
    monthly_table[f"TOTAL_{name.upper()}"] = df2["PREV_TOTAL_MARCHE"].values

monthly_table = monthly_table.sort_values("Date").reset_index(drop=True)

# S1 totals
monthly_table["YM"] = monthly_table["Date"].dt.strftime("%Y-%m")
s1_table = monthly_table[monthly_table["Date"].dt.month.isin([1,2,3,4,5,6])]
s1_summary = {}
for name, _ in dfs.items():
    s1_summary[name] = float(s1_table[f"TOTAL_{name.upper()}"].sum())

rows = []
for name, total in s1_summary.items():
    rows.append({"scenario": name, "S1_2026_TOTAL": int(round(total))})

df_summary = pd.DataFrame(rows)
baseline_total = s1_summary.get("baseline", None)
if baseline_total is not None:
    df_summary["DIFF_FROM_BASELINE_ABS"] = df_summary["S1_2026_TOTAL"] - int(round(baseline_total))
    df_summary["DIFF_FROM_BASELINE_PCT"] = (df_summary["DIFF_FROM_BASELINE_ABS"] / float(baseline_total)) * 100.0

out_csv = os.path.join(project_root, "rapport_ARTES_s1_2026_by_scenario.csv")
monthly_out = os.path.join(project_root, "rapport_ARTES_monthly_by_scenario.csv")
png_out = os.path.join(project_root, "rapport_ARTES_s1_2026_by_scenario.png")
txt_out = os.path.join(project_root, "rapport_ARTES_s1_2026_summary.txt")

monthly_table.to_csv(monthly_out, index=False)
df_summary.to_csv(out_csv, index=False)

# Plot
plt.figure(figsize=(9, 5))
for name in dfs.keys():
    plt.plot(monthly_table["Date"], monthly_table[f"TOTAL_{name.upper()}"], label=name)
plt.title("S1 2026 - Total Market Forecast by Scenario")
plt.xlabel("Month")
plt.ylabel("Projected Monthly Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(png_out, dpi=200)

# Text summary
with open(txt_out, "w", encoding="utf-8") as f:
    f.write("Rapport ARTES — Prévisions S1 2026 par scénario\n")
    f.write("\nMéthodologie: ensemble SARIMAX + Prophet + XGBoost (shares). Scénarios macro fournis dans 'donnees_externes_tunisie.xlsx'.\n\n")
    for _, r in df_summary.iterrows():
        f.write(f"- Scénario {r['scenario'].upper()}: S1 2026 total = {int(r['S1_2026_TOTAL']):,} unités")
        if baseline_total is not None and r['scenario'] != 'baseline':
            f.write(f" ({r['DIFF_FROM_BASELINE_ABS']:+,} vs baseline, {r['DIFF_FROM_BASELINE_PCT']:+.1f}%)")
        f.write("\n")
    f.write("\nRemarques: Ces résultats dépendent fortement des hypothèses macro. Voir CSVs et graphique pour détails mensuels.\n")

print(f"Wrote report CSV: {out_csv}")
print(f"Wrote monthly CSV: {monthly_out}")
print(f"Wrote PNG: {png_out}")
print(f"Wrote textual summary: {txt_out}")
