"""Load Keysight B1500 EasyEXPERT (PBTI / IvSweep) CSV exports and expose the
SMU-measured Id-Vg curve as a FefetSweep, so the reference-curve model gets
accurate current levels (true subthreshold floor ~1e-13 A) instead of the
integrator/ADC back-solved currents (floor ~1e-8 A) from the DAC sweeps.

Each file holds several Id-Vg blocks measured at increasing BTI stress time
(TimeList 0,10,...,3600 s). block=0 is the fresh (t=0) sweep — use that as the
reference unless you specifically want a stressed state.
"""
from __future__ import annotations

import hashlib
import os
import re
import warnings
from dataclasses import dataclass

import numpy as np

from h_aimdc_sim.fefet_measure_io import FefetSweep

# SMU subthreshold noise floor: readings below this |Id| are indistinguishable
# from leakage/offset and are clamped so log10 stays finite.
SMU_I_CLAMP = 1e-14


@dataclass(frozen=True)
class SmuIdVg:
    label: str          # e.g. "fefet_vth_1.0"
    block: int          # stress-time block index (0 = fresh)
    v_g: np.ndarray
    i_d: np.ndarray     # |Id|, A
    v_ds: float
    source_path: str
    source_sha256: str
    measured_at: str


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_idvg_blocks(path: str):
    """Yield (vg, vd, idd) arrays for each 'DataName, Vgate ...' block."""
    vg: list[float] = []
    vd: list[float] = []
    idd: list[float] = []
    in_blk = False
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("DataName, Vgate"):
                if in_blk and vg:
                    yield np.array(vg), np.array(vd), np.array(idd)
                vg, vd, idd, in_blk = [], [], [], True
                continue
            if in_blk:
                if line.startswith("DataValue,"):
                    p = [x.strip() for x in line.split(",")]
                    try:
                        vg.append(float(p[1]))
                        vd.append(float(p[2]))
                        idd.append(float(p[3]))
                    except (ValueError, IndexError):
                        pass
                elif line.strip() and not line.startswith("DataValue"):
                    yield np.array(vg), np.array(vd), np.array(idd)
                    vg, vd, idd, in_blk = [], [], [], False
        if in_blk and vg:
            yield np.array(vg), np.array(vd), np.array(idd)


def load_smu_idvg(path: str, *, block: int = 0) -> SmuIdVg:
    blocks = list(_iter_idvg_blocks(path))
    if not blocks:
        raise ValueError(f"No Id-Vg (Vgate) block found in {os.path.basename(path)}")
    if block >= len(blocks):
        raise IndexError(f"{os.path.basename(path)}: requested block {block} "
                         f"but only {len(blocks)} present")
    vg, vd, idd = blocks[block]
    order = np.argsort(vg)
    vg, vd, idd = vg[order], vd[order], idd[order]
    i_abs = np.maximum(np.abs(idd), SMU_I_CLAMP)
    vds = float(np.median(vd))
    m = re.search(r"(fefet_vth_[0-9.]+)", os.path.basename(path))
    label = m.group(1) if m else os.path.basename(path)
    ts = re.search(r";\s*([0-9_/ :APM]+)\]", os.path.basename(path))
    return SmuIdVg(
        label=label, block=block, v_g=vg.astype(float), i_d=i_abs.astype(float),
        v_ds=vds, source_path=os.path.abspath(path),
        source_sha256=_file_sha256(path),
        measured_at=(ts.group(1).strip() if ts else ""),
    )


def smu_to_sweep(smu: SmuIdVg, *, code: int = 300) -> FefetSweep:
    """Adapt an SMU Id-Vg into a FefetSweep so the existing reference-curve
    fitter (fefet_ref_model) can consume it unchanged."""
    floor_mask = smu.v_g < -3.5
    i_floor = float(np.median(smu.i_d[floor_mask])) if np.any(floor_mask) \
        else float(np.min(smu.i_d))
    return FefetSweep(
        code=int(code),
        v_g=smu.v_g.astype(np.float64, copy=False),
        i_d=smu.i_d.astype(np.float64, copy=False),
        v_ds=float(smu.v_ds),
        i_floor=i_floor,
        rel_noise=0.05,  # SMU per-point noise is small; nominal placeholder
        source_path=smu.source_path,
        source_sha256=smu.source_sha256,
        measured_at=smu.measured_at,
    )


__all__ = ["SmuIdVg", "load_smu_idvg", "smu_to_sweep", "SMU_I_CLAMP"]
