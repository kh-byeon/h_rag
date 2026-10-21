#!/usr/bin/env python3
"""Moved: the publication plot script now lives in RAG_MFMIS.

    cd c:\\Users\\khb\\codes\\RAG_MFMIS
    python plot_q1000_figures.py

This file is a pointer so an old working directory does not silently write
figures next to the experiment outputs.
"""
from __future__ import annotations

import runpy
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "RAG_MFMIS" / "plot_q1000_figures.py"
if not TARGET.is_file():
    raise FileNotFoundError(f"canonical plot script not found: {TARGET}")
print(f"forwarding to {TARGET}")
runpy.run_path(str(TARGET), run_name="__main__")
