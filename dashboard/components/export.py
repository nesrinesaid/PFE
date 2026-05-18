from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import streamlit as st
from fpdf import FPDF

from . import auth


def export_dataframe_excel(df: pd.DataFrame, filename: str = "export.xlsx"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    st.download_button("Télécharger Excel", data=buffer.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def export_dataframes_excel(dataframes: Dict[str, pd.DataFrame], filename: str = "artes_export.xlsx"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in dataframes.items():
            safe_name = str(sheet_name)[:31] or "data"
            df.to_excel(writer, index=False, sheet_name=safe_name)
    st.download_button("📥 Télécharger Excel", data=buffer.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def export_figure_png(fig, filename: str = "chart.png"):
    try:
        img_bytes = fig.to_image(format="png")
        st.download_button("Télécharger PNG", data=img_bytes, file_name=filename, mime="image/png")
    except Exception as e:
        st.error(f"Impossible d'exporter l'image: {e}")


def figure_png_bytes(fig) -> bytes:
    return fig.to_image(format="png")


def _logo_bytes() -> bytes | None:
    candidates = [
        Path(__file__).resolve().parents[2] / "assets" / "logo_artes.png",
        Path(__file__).resolve().parents[2] / "dashboard" / "assets" / "logo_artes.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_bytes()
    return None


def _find_unicode_font_pair() -> tuple[Path, Path | None] | None:
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    pairs = [
        (windows_fonts / "arial.ttf", windows_fonts / "arialbd.ttf"),
        (windows_fonts / "calibri.ttf", windows_fonts / "calibrib.ttf"),
        (windows_fonts / "verdana.ttf", windows_fonts / "verdanab.ttf"),
        (windows_fonts / "tahoma.ttf", windows_fonts / "tahomabd.ttf"),
        (windows_fonts / "segoeui.ttf", windows_fonts / "seguisb.ttf"),
        (windows_fonts / "DejaVuSans.ttf", windows_fonts / "DejaVuSans-Bold.ttf"),
        (windows_fonts / "LiberationSans-Regular.ttf", windows_fonts / "LiberationSans-Bold.ttf"),
    ]
    for regular, bold in pairs:
        if regular.exists():
            return regular, bold if bold.exists() else None
    return None


def _set_pdf_font(pdf: FPDF, size: int = 11, bold: bool = False):
    font_pair = _find_unicode_font_pair()
    if font_pair:
        regular_path, bold_path = font_pair
        font_family = "ARTESUnicode"
        if font_family not in getattr(pdf, "fonts", {}):
            # Register the regular face first, then the bold face when available.
            pdf.add_font(font_family, "", str(regular_path), uni=True)
            if bold_path is not None:
                pdf.add_font(font_family, "B", str(bold_path), uni=True)
        pdf.set_font(font_family, "B" if bold and bold_path is not None else "", size)
        return font_family
    pdf.set_font("Helvetica", "B" if bold else "", size)
    return "Helvetica"


def build_pdf_report(title: str, summary_lines: Iterable[str] | None = None, figures: Dict[str, object] | None = None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    logo = _logo_bytes()
    if logo:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_logo:
            tmp_logo.write(logo)
            tmp_logo_path = tmp_logo.name
        pdf.image(tmp_logo_path, x=10, y=8, w=40)
        try:
            Path(tmp_logo_path).unlink(missing_ok=True)
        except Exception:
            pass

    _set_pdf_font(pdf, size=16, bold=True)
    pdf.ln(18)
    pdf.cell(0, 10, str(title), ln=True)
    _set_pdf_font(pdf, size=11, bold=False)
    pdf.ln(2)
    if summary_lines:
        for line in summary_lines:
            pdf.multi_cell(0, 6, f"- {line}")

    if figures:
        for fig_title, fig in figures.items():
            try:
                pdf.add_page()
                _set_pdf_font(pdf, size=13, bold=True)
                pdf.cell(0, 8, str(fig_title), ln=True)
                img_bytes = figure_png_bytes(fig)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    tmp_img.write(img_bytes)
                    tmp_img_path = tmp_img.name
                pdf.image(tmp_img_path, x=10, w=190)
                try:
                    Path(tmp_img_path).unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as exc:
                pdf.add_page()
                _set_pdf_font(pdf, size=11, bold=False)
                pdf.multi_cell(0, 6, f"{fig_title}: export impossible ({exc})")

    return bytes(pdf.output(dest="S"))


def render_export_panel(title: str, prefix: str, dataframes: Dict[str, pd.DataFrame] | None = None, figures: Dict[str, object] | None = None, summary_lines: Iterable[str] | None = None):
    figures = {k: v for k, v in (figures or {}).items() if v is not None}
    with st.expander("📥 Exporter", expanded=False):
        st.caption("Téléchargez les résultats de cette page au format PDF, Excel ou PNG.")
        if st.button("Créer le PDF", key=f"{prefix}_create_pdf", use_container_width=True):
            st.session_state[f"{prefix}_pdf_bytes"] = build_pdf_report(title, summary_lines=summary_lines, figures=figures)
        pdf_bytes = st.session_state.get(f"{prefix}_pdf_bytes")
        if pdf_bytes:
            st.download_button("Télécharger le PDF", data=pdf_bytes, file_name=f"{prefix}.pdf", mime="application/pdf", use_container_width=True, key=f"{prefix}_download_pdf")
        if dataframes:
            excel_bytes = io.BytesIO()
            with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
                for sheet_name, df in dataframes.items():
                    df.to_excel(writer, index=False, sheet_name=str(sheet_name)[:31] or "data")
            st.download_button("Télécharger Excel", data=excel_bytes.getvalue(), file_name=f"{prefix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key=f"{prefix}_download_xlsx")
        if figures:
            for fig_name, fig in figures.items():
                try:
                    st.download_button(
                        f"PNG: {fig_name}",
                        data=figure_png_bytes(fig),
                        file_name=f"{prefix}_{fig_name}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"{prefix}_png_{fig_name}",
                    )
                except Exception as exc:
                    st.error(f"PNG indisponible pour {fig_name} : {exc}")
