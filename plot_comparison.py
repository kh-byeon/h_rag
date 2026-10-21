#!/usr/bin/env python3
"""
Plot simulated I-V vs measured I-V.

Usage:
  python plot_comparison.py --sim sim_igzo.prn --meas igzo_measure.csv --out iv_comparison.png
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _coerce_float(value: str):
    try:
        return float(value.strip())
    except Exception:
        return None


def _read_table_auto(path: str) -> Tuple[List[str], List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"Empty file: {path}")

    header_idx = 0
    for i, ln in enumerate(lines):
        if any(ch.isalpha() for ch in ln):
            header_idx = i
            break
    header_line = lines[header_idx]
    body_lines = lines[header_idx + 1 :]

    if "," in header_line:
        headers = [h.strip() for h in header_line.split(",")]
        rows: List[List[float]] = []
        for ln in body_lines:
            parts = [p.strip() for p in ln.split(",")]
            vals = [_coerce_float(p) for p in parts]
            if all(v is not None for v in vals) and len(vals) == len(headers):
                rows.append([float(v) for v in vals if v is not None])
        return headers, rows

    headers = re.split(r"\s+", header_line)
    rows = []
    for ln in body_lines:
        parts = re.split(r"\s+", ln)
        vals = [_coerce_float(p) for p in parts]
        if all(v is not None for v in vals) and len(vals) == len(headers):
            rows.append([float(v) for v in vals if v is not None])
    return headers, rows


def _pick_columns(headers: List[str], rows: List[List[float]], kind: str) -> Tuple[List[float], List[float]]:
    if not rows:
        raise ValueError(f"No numeric rows found in {kind} data.")
    ncols = len(headers)
    cols: Dict[str, List[float]] = {headers[i]: [r[i] for r in rows] for i in range(ncols)}
    norm_to_raw = {_normalize(h): h for h in headers}

    if kind == "sim":
        x_candidates = ["v1", "v(1)", "v", "voltage", "vgs", "vg"]
        y_candidates = ["in1", "i(n1)", "id", "i", "current", "ids"]
    else:
        x_candidates = ["voltage", "v", "vgs", "vg", "index", "vexpected"]
        y_candidates = ["current", "i", "id", "ids", "imeasured", "vexpected"]

    x_key = None
    for c in x_candidates:
        if c in norm_to_raw:
            x_key = norm_to_raw[c]
            break
    y_key = None
    for c in y_candidates:
        if c in norm_to_raw:
            y_key = norm_to_raw[c]
            break

    if x_key is None:
        x_key = headers[0]
    if y_key is None:
        y_key = headers[1] if len(headers) > 1 else headers[0]

    return cols[x_key], cols[y_key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simulated and measured I-V curves.")
    parser.add_argument("--sim", default="sim_igzo.prn", help="Xyce output file (.prn or .csv).")
    parser.add_argument("--meas", default="igzo_measure.csv", help="Measured CSV file.")
    parser.add_argument("--out", default="iv_comparison.png", help="Output plot path.")
    args = parser.parse_args()

    sim_headers, sim_rows = _read_table_auto(args.sim)
    sim_v, sim_i = _pick_columns(sim_headers, sim_rows, kind="sim")

    meas_headers, meas_rows = _read_table_auto(args.meas)
    meas_v, meas_i = _pick_columns(meas_headers, meas_rows, kind="meas")

    plt.figure(figsize=(8, 5))
    plt.plot(sim_v, sim_i, label="Simulated I-V", linewidth=2)
    plt.plot(meas_v, meas_i, "o", label="Measured", markersize=4, alpha=0.8)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("IGZO TFT I-V Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[plot] saved {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
