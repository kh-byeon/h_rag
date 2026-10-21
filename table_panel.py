"""Draw n=3500 paper tables into a caller-owned axes (no figure, no save).

Statistics are imported from scripts.make_paper_tables; this module only typesets.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.make_paper_tables import QUANTS, RUN, generation, retrieval

MINUS = "\u2212"

# Shared 5-column grid in axes coordinates. Metric is left-aligned at [0];
# columns [1:] are right-aligned numeric (retrieval) or Retriever + numerics
# (quantization). Callers must pass the same tuple to both blocks.
DEFAULT_COL_X = (0.02, 0.50, 0.64, 0.78, 0.97)

_CACHE: dict = {}


def _data():
    if not _CACHE:
        _CACHE["retr"] = retrieval()
        _CACHE["gen"] = generation()
    return _CACHE["retr"], _CACHE["gen"]


def _plain(name: str) -> str:
    return (
        name.replace(r" $\uparrow$", " \u2191")
        .replace(r" $\downarrow$", " \u2193")
        .replace("$\\uparrow$", "\u2191")
        .replace("$\\downarrow$", "\u2193")
    )


def _fmt(v, dp=3) -> str:
    return f"{v:.{dp}f}".replace("-", MINUS)


def _signed(v, dp=3) -> str:
    return f"{v:+.{dp}f}".replace("-", MINUS)


def _text(ax, x, y, s, *, ha, va="center", fontsize, weight="normal",
          style="normal", color="black"):
    ax.text(
        x, y, s,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize,
        fontweight=weight,
        fontstyle=style,
        color=color,
        clip_on=False,
    )


def _hrule(ax, y, lw, x0=0.0, x1=1.0):
    ax.plot(
        [x0, x1], [y, y],
        transform=ax.transAxes,
        clip_on=False,
        color="black",
        lw=lw,
        solid_capstyle="butt",
    )


def _pack_ys(n_rows: int, y_top: float, y_bot: float) -> list[float]:
    if n_rows == 1:
        return [0.5 * (y_top + y_bot)]
    step = (y_top - y_bot) / (n_rows - 1)
    return [y_top - i * step for i in range(n_rows)]


def _draw_retrieval(ax, *, fontsize, header_fontsize, thick, thin, col_x, title):
    coverage, diversity, _rs = _data()[0]

    rows: list[tuple] = [("header", None)]
    rows.append(("section", "Evidence coverage"))
    for item in coverage:
        rows.append(("cov", item))
    rows.append(("skip", None))
    rows.append(("section", "Diversity of the retrieved set"))
    for item in diversity:
        rows.append(("div", item))

    y_top = 0.86 if title else 0.92
    y_bot = 0.16
    ys = _pack_ys(len(rows), y_top, y_bot)
    y_map = {i: ys[i] for i in range(len(rows))}

    if title:
        _text(ax, col_x[0], 0.97, title, ha="left", va="top",
              fontsize=header_fontsize, weight="bold")

    # Header rule (thick), header text, midrule (thin).
    y_head = y_map[0]
    _hrule(ax, y_head + 0.045, thick)
    headers = ("Metric", "L2", "MMR", "\u0394", "\u0394 (%)")
    aligns = ("left", "right", "right", "right", "right")
    for x, h, ha in zip(col_x, headers, aligns):
        _text(ax, x, y_head, h, ha=ha, fontsize=header_fontsize, weight="bold")
    _hrule(ax, y_head - 0.035, thin)

    last_data_y = y_head
    for i, (kind, payload) in enumerate(rows):
        y = y_map[i]
        if kind == "header":
            continue
        if kind == "skip":
            continue
        if kind == "section":
            _text(ax, col_x[0], y, payload, ha="left", fontsize=fontsize,
                  style="italic")
            continue
        name, a, b, bold = payload
        d = b - a
        pct = 100.0 * d / a
        label = _plain(name)
        if kind == "div":
            label = label + "*"
        wt_mmr = "bold" if bold else "normal"
        _text(ax, col_x[0], y, label, ha="left", fontsize=fontsize)
        _text(ax, col_x[1], y, _fmt(a), ha="right", fontsize=fontsize)
        _text(ax, col_x[2], y, _fmt(b), ha="right", fontsize=fontsize,
              weight=wt_mmr)
        _text(ax, col_x[3], y, _signed(d), ha="right", fontsize=fontsize,
              weight=wt_mmr)
        _text(ax, col_x[4], y, _signed(pct, 1), ha="right", fontsize=fontsize,
              weight=wt_mmr)
        last_data_y = y

    _hrule(ax, last_data_y - 0.040, thick)
    _text(
        ax, col_x[0], 0.045,
        "*Corpus aggregate; no per-query interval available.",
        ha="left", va="bottom", fontsize=6, color="0.45",
    )


def _draw_quantization(ax, *, fontsize, header_fontsize, thick, thin, col_x, title):
    gen, _did = _data()[1]
    q0, q1 = QUANTS[0], QUANTS[-1]

    # header + 2 rows per metric, with a skip between metrics
    n_metrics = len(gen)
    n_rows = 1 + n_metrics * 2 + max(n_metrics - 1, 0)
    y_top = 0.86 if title else 0.92
    y_bot = 0.10
    ys = _pack_ys(n_rows, y_top, y_bot)

    if title:
        _text(ax, col_x[0], 0.97, title, ha="left", va="top",
              fontsize=header_fontsize, weight="bold")

    y_head = ys[0]
    _hrule(ax, y_head + 0.050, thick)
    headers = ("Metric", "Retriever", q0, q1, "Change (%)")
    aligns = ("left", "left", "right", "right", "right")
    for x, h, ha in zip(col_x, headers, aligns):
        _text(ax, x, y_head, h, ha=ha, fontsize=header_fontsize, weight="bold")
    _hrule(ax, y_head - 0.040, thin)

    row_i = 1
    last_data_y = y_head
    for m_i, (label, arms) in enumerate(gen):
        y_l2 = ys[row_i]
        y_mmr = ys[row_i + 1]
        y_mid = 0.5 * (y_l2 + y_mmr)
        _text(ax, col_x[0], y_mid, label, ha="left", fontsize=fontsize)
        sig = arms.get("_sig", {})
        for arm, y in (("L2", y_l2), ("MMR", y_mmr)):
            _text(ax, col_x[1], y, arm, ha="left", fontsize=fontsize)
            v0 = arms[arm]["absv"][q0]
            v1 = arms[arm]["absv"][q1]
            rel = arms[arm]["rel"][q1]
            w0 = "bold" if (arm == "MMR" and sig.get(q0)) else "normal"
            w1 = "bold" if (arm == "MMR" and sig.get(q1)) else "normal"
            _text(ax, col_x[2], y, _fmt(v0), ha="right", fontsize=fontsize,
                  weight=w0)
            _text(ax, col_x[3], y, _fmt(v1), ha="right", fontsize=fontsize,
                  weight=w1)
            _text(ax, col_x[4], y, _signed(rel, 2), ha="right", fontsize=fontsize)
            last_data_y = y
        row_i += 2
        if m_i < n_metrics - 1:
            row_i += 1  # skip / addlinespace

    _hrule(ax, last_data_y - 0.045, thick)


def draw_table_panel(ax, block, *, fontsize=7, header_fontsize=7,
                     rule_lw=(1.0, 0.5), col_x=None, title=None):
    """Render one tabular block into `ax`. Creates no figure, saves nothing.

    ax     : an axes the caller owns (from GridSpec). The function calls
             ax.set_axis_off() and draws in axes coordinates (0-1), so the panel
             scales with whatever cell it is given.
    block  : "retrieval" or "quantization"
    col_x  : optional tuple of 5 x positions in axes coords; pass the SAME tuple for
             both blocks so the two panels share a column grid.
    returns: the col_x actually used, so the caller can reuse it.
    """
    if col_x is None:
        col_x = DEFAULT_COL_X
    if len(col_x) != 5:
        raise ValueError(f"col_x must have 5 positions, got {len(col_x)}")
    col_x = tuple(float(x) for x in col_x)
    thick, thin = rule_lw

    ax.set_axis_off()
    if block == "retrieval":
        _draw_retrieval(
            ax, fontsize=fontsize, header_fontsize=header_fontsize,
            thick=thick, thin=thin, col_x=col_x, title=title,
        )
    elif block == "quantization":
        _draw_quantization(
            ax, fontsize=fontsize, header_fontsize=header_fontsize,
            thick=thick, thin=thin, col_x=col_x, title=title,
        )
    else:
        raise ValueError(f"unknown block {block!r}; expected 'retrieval' or 'quantization'")
    return col_x
