import os
import pandas as pd

"""Create three example macro scenario sheets in data/donnees_externes_tunisie.xlsx:
   - baseline
   - optimiste
   - prudent

Each sheet will contain monthly Date, PIB_INDEX and INFLATION columns from 2019-01 to 2026-06.
This is a helper to populate the workbook with sensible example scenarios for step6.
"""


def make_series(dates):
    # baseline: gentle growth in PIB index and stable inflation
    baseline_pib = 100.0 + (pd.Series(range(len(dates))) * 0.5).values
    baseline_infl = 6.0 + (pd.Series(range(len(dates))) * 0.01).values

    # optimiste: stronger growth, lower inflation
    opt_pib = baseline_pib * 1.03
    opt_infl = baseline_infl * 0.9

    # prudent: weaker growth, higher inflation
    pru_pib = baseline_pib * 0.98
    pru_infl = baseline_infl * 1.15

    return (
        (baseline_pib, baseline_infl),
        (opt_pib, opt_infl),
        (pru_pib, pru_infl),
    )


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(project_root, "data", "donnees_externes_tunisie.xlsx")
    dates = pd.date_range("2019-01-01", "2026-06-01", freq="MS")
    (b_pib, b_inf), (o_pib, o_inf), (p_pib, p_inf) = make_series(dates)

    df_base = pd.DataFrame({"Date": dates, "PIB_INDEX": b_pib, "INFLATION": b_inf})
    df_opt = pd.DataFrame({"Date": dates, "PIB_INDEX": o_pib, "INFLATION": o_inf})
    df_pru = pd.DataFrame({"Date": dates, "PIB_INDEX": p_pib, "INFLATION": p_inf})

    # Write three sheets
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a" if os.path.exists(out_path) else "w") as writer:
        df_base.to_excel(writer, sheet_name="baseline", index=False)
        df_opt.to_excel(writer, sheet_name="optimiste", index=False)
        df_pru.to_excel(writer, sheet_name="prudent", index=False)

    print(f"Wrote scenario sheets to {out_path}")


if __name__ == "__main__":
    main()
