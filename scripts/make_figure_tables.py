#!/usr/bin/env python3
"""Nature double-column composite figure with n=3500 table panels.

    python scripts/make_figure_tables.py

Writes figures/fig_main.pdf (live text) and figures/fig_main.png (review copy).
Plot panels that do not yet accept an `ax` are left empty with a TODO comment.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.make_paper_tables import QUANTS
from table_panel import _data, draw_table_panel

OUT_DIR = _REPO / "figures"
W_MM = 180.0
H_MM = 150.0
TICK = 7  # table body matches plot-panel tick labels


def _assert_close(got, exp, name, tol=1e-3):
    if abs(got - exp) > tol:
        raise AssertionError(f"{name}: got {got}, expected {exp} (tol {tol})")


def _assert_stats(cov, div, gen):
    fs = next(r for r in cov if r[0] == "Full support @5")
    _assert_close(fs[1], 0.565, "retrieval Full support @5 L2")
    _assert_close(fs[2], 0.596, "retrieval Full support @5 MMR")
    _assert_close(fs[2] - fs[1], 0.031, "retrieval Full support @5 Δ")

    br = next(r for r in div if "Centroid breadth" in r[0])
    _assert_close(br[1], 2.435, "retrieval Centroid breadth L2")
    _assert_close(br[2], 2.669, "retrieval Centroid breadth MMR")

    q0, q1 = QUANTS[0], QUANTS[-1]
    f1_arms = next(arms for label, arms in gen if label == "Answer F1")
    _assert_close(f1_arms["L2"]["absv"][q0], 0.581, "generation Answer F1 L2 int8")
    _assert_close(f1_arms["L2"]["absv"][q1], 0.561, "generation Answer F1 L2 int4")
    _assert_close(round(f1_arms["L2"]["rel"][q1], 2), -3.41, "generation Answer F1 L2 change %")

    fa_arms = next(arms for label, arms in gen if label == "Faithfulness")
    _assert_close(fa_arms["MMR"]["absv"][q0], 0.907, "generation Faithfulness MMR int8")
    _assert_close(fa_arms["MMR"]["absv"][q1], 0.838, "generation Faithfulness MMR int4")
    _assert_close(round(fa_arms["MMR"]["rel"][q1], 2), -7.63, "generation Faithfulness MMR change %")
    if fa_arms["_sig"][q0] or fa_arms["_sig"][q1]:
        raise AssertionError(
            f"generation Faithfulness _sig must be False at BOTH precisions; "
            f"got {fa_arms['_sig']}"
        )


def _panel_letter(ax, letter: str):
    ax.text(
        -0.02, 1.02, letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
        ha="right",
        clip_on=False,
    )


def _sci_p(p: float) -> str:
    if p >= 1e-3:
        return f"{p:.3f}"
    e = int(f"{p:.1e}".split("e")[1])
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{p / 10 ** e:.1f} \u00d7 10{str(e).translate(sup)}"


def _legend_fragment(rs, did) -> str:
    f1 = did["Answer F1"]
    p_str = _sci_p(rs["p"])
    return (
        "Figure legend. Centre values are means over n = 3500 HotpotQA queries. "
        "Intervals (not shown in the table) are 95% CIs from a paired bootstrap "
        "over queries, 20,000 resamples. Full support @5 was compared with an "
        f"exact McNemar test (MMR-only {rs['only_mmr']} vs L2-only {rs['only_l2']}; "
        f"P = {p_str}). The difference-in-differences of int4 versus int8 Answer F1 "
        f"between MMR and L2 is not resolved "
        f"(DID {f1['mu']:+.4f}, 95% CI [{f1['lo']:+.4f}, {f1['hi']:+.4f}], "
        f"P = {f1['p']:.2f})."
    )


def _pdf_mediabox_mm(path: Path) -> tuple[float, float]:
    raw = path.read_bytes()
    # First MediaBox in the file is the page box, in PDF points (1/72 in).
    import re
    m = re.search(rb"/MediaBox\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]", raw)
    if not m:
        raise RuntimeError(f"no MediaBox in {path}")
    x0, y0, x1, y1 = map(float, m.groups())
    w_mm = (x1 - x0) / 72.0 * 25.4
    h_mm = (y1 - y0) / 72.0 * 25.4
    return w_mm, h_mm


def _pdftotext(path: Path) -> str:
    exe = shutil.which("pdftotext")
    if exe:
        r = subprocess.run([exe, str(path), "-"], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            return r.stdout
        print(f"pdftotext failed ({r.returncode}): {r.stderr}", file=sys.stderr)
    # Fallback: pypdf / PyPDF2
    try:
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    except Exception as exc:
        raise RuntimeError(
            "pdftotext is not on PATH and pypdf/PyPDF2 could not extract text"
        ) from exc


def main():
    (coverage, diversity, rs), (gen, did) = _data()
    _assert_stats(coverage, diversity, gen)
    print("asserts ok")

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["font.size"] = TICK
    rcParams["axes.labelsize"] = TICK
    rcParams["xtick.labelsize"] = TICK
    rcParams["ytick.labelsize"] = TICK

    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.05, 1.40, 1.10],
        hspace=0.32,
        wspace=0.18,
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.05,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    # TODO: retrieval plot — no ax-taking function exists yet
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.tick_params(labelsize=TICK)

    ax_b = fig.add_subplot(gs[0, 1])
    # TODO: generation plot — no ax-taking function exists yet
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.tick_params(labelsize=TICK)

    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])

    col_x = draw_table_panel(
        ax_c, "retrieval",
        fontsize=TICK, header_fontsize=TICK, title="Retrieval",
    )
    draw_table_panel(
        ax_d, "quantization",
        fontsize=TICK, header_fontsize=TICK, col_x=col_x, title="Quantization",
    )

    _panel_letter(ax_a, "a")
    _panel_letter(ax_b, "b")
    _panel_letter(ax_c, "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "fig_main.pdf"
    png_path = OUT_DIR / "fig_main.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    w_mm, h_mm = _pdf_mediabox_mm(pdf_path)
    print(f"saved {pdf_path.relative_to(_REPO)}  mediabox {w_mm:.2f} x {h_mm:.2f} mm")
    print(f"saved {png_path.relative_to(_REPO)}")
    if abs(w_mm - W_MM) > 0.05:
        raise AssertionError(f"figure width {w_mm:.3f} mm != {W_MM} mm (bbox_inches tight?)")

    text = _pdftotext(pdf_path)
    needles = ("0.565", "0.596", "2.435", "2.669", "0.581", "0.907", "Full support")
    missing = [n for n in needles if n not in text.replace(" ", "")]
    # Allow whitespace inserted by extractors.
    compact = text.replace(" ", "").replace("\n", "")
    missing = [n for n in needles if n.replace(" ", "") not in compact]
    print("--- pdftotext ---")
    print(text if text.strip() else "[EMPTY]")
    print("--- end pdftotext ---")
    if not text.strip() or missing:
        raise SystemExit(
            f"PDF text is missing live table numbers {missing}. "
            "Text was rasterised or extraction failed."
        )
    print("pdftotext: table numbers are live selectable text")
    print()
    print(_legend_fragment(rs, did))


if __name__ == "__main__":
    main()
