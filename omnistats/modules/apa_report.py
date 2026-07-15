"""
omnistats/modules/apa_report.py
────────────────────────────────
APA 7th Edition Word document report generator.
Migrated & extended from lpa_analysis/step6_apa_tables.py.

Generates a single Word document containing:
  Table 1 — LPA Model Fit Statistics
  Table 2 — Profile Means, SDs, Welch ANOVA
  Table 3 — Chi-Square Tests for Demographics
  Table 4 — Profile Membership Summary
  Table 5 — A/B Test Results (if available)
  Table 6 — Causal Inference Results (if available)
"""
import os
import sys
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, INDICATOR_COLS, N_PROFILES


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _set_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        val = kwargs.get(side, "none")
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), val)
        if val != "none":
            b.set(qn("w:sz"), "6")
            b.set(qn("w:space"), "0")
            b.set(qn("w:color"), "000000")
        tcBorders.append(b)
    tcPr.append(tcBorders)


def _clear_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    tblBorders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "none")
        tblBorders.append(b)
    tblPr.append(tblBorders)


def _hline(row, pos="bottom"):
    for cell in row.cells:
        _set_border(cell, **{pos: "single"})


def _cell(cell, text, bold=False, italic=False, align="left", size=11):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    para = cell.paragraphs[0]
    para.clear()
    para.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT,
                      "center": WD_ALIGN_PARAGRAPH.CENTER,
                      "right": WD_ALIGN_PARAGRAPH.RIGHT}.get(align, WD_ALIGN_PARAGRAPH.LEFT)
    run = para.add_run(str(text))
    run.bold = bold; run.italic = italic
    run.font.name = "Times New Roman"; run.font.size = Pt(size)


def _title(doc, text: str, num: int):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"Table {num}")
    r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(12)
    p.add_run("\n")
    r2 = p.add_run(text)
    r2.italic = True; r2.font.name = "Times New Roman"; r2.font.size = Pt(12)


def _note(doc, text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True; r.font.size = Pt(10); r.font.name = "Times New Roman"


def _table(doc, headers, rows, col_widths=None):
    ncols = len(headers)
    tbl   = doc.add_table(rows=1 + len(rows), cols=ncols)
    tbl.style = "Table Grid"
    _clear_table_borders(tbl)

    hrow = tbl.rows[0]
    for j, h in enumerate(headers):
        _cell(hrow.cells[j], h, bold=True, align="left" if j == 0 else "center")
    _hline(hrow, "top"); _hline(hrow, "bottom")

    for i, row_data in enumerate(rows):
        drow = tbl.rows[i + 1]
        for j, val in enumerate(row_data):
            _cell(drow.cells[j], val, align="left" if j == 0 else "right")

    _hline(tbl.rows[-1], "bottom")

    if col_widths:
        for ci, w in enumerate(col_widths):
            for row in tbl.rows:
                row.cells[ci].width = Inches(w)

    doc.add_paragraph()


def _fmt(val, dec=2):
    if pd.isna(val):
        return "—"
    try:
        return f"{float(val):.{dec}f}"
    except Exception:
        return str(val)


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_report(verbose: bool = True) -> None:
    """Build and save the full APA report to outputs/apa_report.docx."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CSVs
    def _read(name):
        p = os.path.join(OUTPUT_DIR, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    fit_df    = _read("lpa_fit_stats.csv")
    prof_df   = _read("lpa_profiles.csv")
    anov_df   = _read("anova_results.csv")
    chi2_df   = _read("chi_square_results.csv")
    ab_df     = _read("ab_test_results.csv")
    caus_df   = _read("causal_results.csv")

    doc   = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"; style.font.size = Pt(12)

    h = doc.add_heading("OmniStats — APA 7th Edition Tables", level=1)
    h.runs[0].font.name = "Times New Roman"
    doc.add_paragraph(
        "Note. Tables generated automatically by OmniStats. "
        "Significance levels: * p < .05, ** p < .01, *** p < .001."
    ).italic = True
    doc.add_page_break()

    tbl_num = 1

    # ── Table 1: LPA Fit Statistics ───────────────────────────────────────────
    _title(doc, "Model Fit Statistics for Latent Profile Analysis (K = 1–6)", tbl_num); tbl_num += 1
    if not fit_df.empty:
        hdrs = ["K", "Log-Likelihood", "AIC", "BIC", "aBIC", "Entropy", "LMR-LRT p"]
        rows = []
        for _, r in fit_df.iterrows():
            lmr = _fmt(r.get("LMR_LRT_p"), 3)
            rows.append([int(r["K"]), _fmt(r["LogLikelihood"], 2), _fmt(r["AIC"], 2),
                         _fmt(r["BIC"], 2), _fmt(r["aBIC"], 2), _fmt(r["Entropy"], 3), lmr])
        _table(doc, hdrs, rows, [0.4, 1.2, 1.0, 1.0, 1.0, 0.85, 0.9])
        _note(doc, "Note. AIC = Akaike Information Criterion; BIC = Bayesian Information Criterion; "
                   "aBIC = Sample-size Adjusted BIC; Entropy = classification entropy (0–1); "
                   "LMR-LRT p = Lo-Mendell-Rubin approximate Likelihood Ratio Test p-value.")
    doc.add_page_break()

    # ── Table 2: Profile Means + ANOVA ───────────────────────────────────────
    _title(doc, f"Profile Indicator Means (SD) and Welch's ANOVA Results (K = {N_PROFILES})", tbl_num); tbl_num += 1
    if not prof_df.empty:
        profiles = sorted(prof_df["Profile"].unique())
        hdrs = ["Indicator"] + [f"Profile {p}" for p in profiles] + ["F", "df1", "df2", "p", "η²"]
        rows = []
        for col in INDICATOR_COLS:
            if col not in prof_df.columns:
                continue
            row = [col]
            for p in profiles:
                grp = prof_df[prof_df["Profile"] == p][col].dropna()
                row.append(f"{grp.mean():.2f} ({grp.std():.2f})")
            if not anov_df.empty and col in anov_df["Indicator"].values:
                ar  = anov_df[anov_df["Indicator"] == col].iloc[0]
                sig = ar.get("sig", "")
                f_str = f"{ar['F_Welch']:.2f}{sig}" if pd.notna(ar.get("F_Welch")) else "—"
                row += [f_str, _fmt(ar["df1"], 0).replace(".00", ""),
                        _fmt(ar["df2"], 1), _fmt(ar["p"], 3), _fmt(ar["eta_squared"], 3)]
            else:
                row += ["—"] * 5
            rows.append(row)
        _table(doc, hdrs, rows)
        _note(doc, "Note. Values are M (SD) for raw scores. F = Welch's F; η² = eta-squared. * p < .05.")
    doc.add_page_break()

    # ── Table 3: Chi-Square ───────────────────────────────────────────────────
    _title(doc, "Chi-Square Tests of Profile Differences on Demographic Variables", tbl_num); tbl_num += 1
    if not chi2_df.empty:
        hdrs = ["Demographic Variable", "χ²", "df", "p", "Cramér's V"]
        rows = []
        for _, r in chi2_df.iterrows():
            rows.append([r["Demographic"],
                         f"{r['chi2']:.2f}{r.get('sig', '')}",
                         str(int(r["df"])),
                         _fmt(r["p"], 3),
                         _fmt(r["Cramers_V"], 3)])
        _table(doc, hdrs, rows, [2.0, 0.9, 0.5, 0.7, 1.1])
        _note(doc, "Note. Cramér's V is the effect size. * p < .05.")
    doc.add_page_break()

    # ── Table 4: Profile Membership ───────────────────────────────────────────
    _title(doc, "Profile Membership Counts and Percentages", tbl_num); tbl_num += 1
    if not prof_df.empty:
        total = len(prof_df)
        hdrs  = ["Profile", "n", "%", "Mean Max Prob."]
        rows  = []
        for p in sorted(prof_df["Profile"].unique()):
            grp  = prof_df[prof_df["Profile"] == p]
            prob = grp["Profile_Max_Prob"].mean() if "Profile_Max_Prob" in grp else np.nan
            rows.append([f"Profile {p}", str(len(grp)),
                         f"{len(grp) / total * 100:.1f}%",
                         f"{prob:.3f}" if not np.isnan(prob) else "—"])
        rows.append(["Total", str(total), "100.0%", "—"])
        _table(doc, hdrs, rows, [1.4, 0.7, 0.7, 1.5])
        _note(doc, "Note. Mean Max Prob. = average posterior probability of the most likely class assignment.")
    doc.add_page_break()

    # ── Table 5: A/B Test Results ─────────────────────────────────────────────
    if not ab_df.empty:
        _title(doc, "A/B Test Results Summary", tbl_num); tbl_num += 1
        cols_ab = [c for c in ["test", "p_value", "significant", "diff", "cohen_d",
                                "lift_pct", "z_stat", "t_stat"] if c in ab_df.columns]
        hdrs = [c.replace("_", " ").title() for c in cols_ab]
        rows = [[str(row[c]) for c in cols_ab] for _, row in ab_df.iterrows()]
        _table(doc, hdrs, rows)
        _note(doc, "Note. Significance at alpha = .05. Cohen's d and lift % reported where applicable.")
        doc.add_page_break()

    # ── Table 6: Causal Inference Results ─────────────────────────────────────
    if not caus_df.empty:
        _title(doc, "Causal Inference Results Summary", tbl_num); tbl_num += 1
        # Standardised schema columns (new modules/causal/ subpackage)
        SCHEMA = ["method", "estimand", "estimate", "se",
                  "ci_lower", "ci_upper", "ci_type", "p_value", "n_obs"]
        caus_df = caus_df.reindex(columns=SCHEMA)
        hdrs = ["Method", "Estimand", "Estimate", "SE",
                "95% CI Lower", "95% CI Upper", "CI Type", "p", "N"]
        rows = []
        for _, row in caus_df.iterrows():
            rows.append([
                str(row["method"]) if pd.notna(row.get("method")) else "—",
                str(row["estimand"]) if pd.notna(row.get("estimand")) else "—",
                _fmt(row.get("estimate"), 4),
                _fmt(row.get("se"), 4),
                _fmt(row.get("ci_lower"), 4),
                _fmt(row.get("ci_upper"), 4),
                str(row["ci_type"]) if pd.notna(row.get("ci_type")) else "—",
                _fmt(row.get("p_value"), 4),
                str(int(row["n_obs"])) if pd.notna(row.get("n_obs")) else "—",
            ])
        _table(doc, hdrs, rows)
        _note(doc,
              "Note. DiD = Callaway & Sant-Anna (2021) staggered ATT(g,t) via differences; "
              "IV = linearmodels IV2SLS (HC3 SEs; Anderson-Rubin CI reported when KP rk-F < 10); "
              "RDD = rdrobust CCT MSE-optimal bandwidth with bias-corrected robust CI and "
              "rddensity manipulation test. "
              "CI Type: doubly_robust = doubly-robust bootstrap; "
              "anderson_rubin = identification-robust CI; robust_bc = bias-corrected robust.")

    out_path = os.path.join(OUTPUT_DIR, "apa_report.docx")
    doc.save(out_path)
    if verbose:
        print(f"[Report] APA Word document saved -> {out_path}")
