#!/usr/bin/env python3
"""
Offline B1 calibration: k maps AIMDC query–candidate distances to software-L2 scale.

    k = median_q { median_i |d_sw| / median_i |d_dev| }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
from numpy.typing import NDArray

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from online_accelerator_device import (  # noqa: E402
    _reconstruct_pq_all,
    load_hw_state_from_dir,
)


def _fit_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(os.path.abspath(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _query_post_matrix(
    raw: NDArray[Any],
    opq: Optional[NDArray[Any]],
) -> NDArray[np.float32]:
    x = np.asarray(raw, dtype=np.float32)
    if opq is None or not isinstance(opq, np.ndarray) or opq.size == 0:
        return cast(NDArray[np.float32], x)
    rot = np.asarray(opq, dtype=np.float32)
    dim = x.shape[1]
    if rot.ndim == 2 and rot.shape[0] == dim:
        return cast(NDArray[np.float32], (x @ rot).astype(np.float32))
    return cast(NDArray[np.float32], x)


def _fit_device_on_pq_recon(
    device_fit_path: str,
    pq_recon: NDArray[Any],
    *,
    scaler_path: str,
    device_reduce: str,
    device_use_log: bool,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
) -> Any:
    from aimdc_device_distance import DeviceDistance, NoiseModel

    reduce_mode: Literal["sum", "mean"] = "mean" if device_reduce == "mean" else "sum"
    dd = DeviceDistance(
        os.path.abspath(device_fit_path),
        scaler_path=os.path.abspath(scaler_path),
        reduce=reduce_mode,
        use_log=bool(device_use_log),
        v_ml=0.0,
        noise=NoiseModel(),
        v_min=v_min,
        v_max=v_max,
    )
    if not os.path.isfile(scaler_path):
        dd.fit_scaler(np.asarray(pq_recon, dtype=np.float64))
        dd.save_scaler(scaler_path)
    return dd


def run_calibration(
    offline_dir: str,
    device_fit_path: str,
    *,
    n_queries: int,
    n_candidates: int,
    seed: int,
    device_reduce: str = "sum",
    device_use_log: bool = True,
    max_pq_rows: int = 50_000,
    device: Any = None,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
) -> Dict[str, Any]:
    hw = load_hw_state_from_dir(os.path.abspath(offline_dir))
    raw = hw.get("raw_embeddings")
    if raw is None or not isinstance(raw, np.ndarray) or raw.ndim != 2:
        raise RuntimeError("hw_state must include raw_embeddings.npy (N, D)")
    lut = hw.get("lut_memory")
    pq_map = hw.get("pq_codes")
    if not isinstance(pq_map, dict):
        pq_map = hw.get("l2_accelerator_2_pq_codes")
    if lut is None or not isinstance(lut, np.ndarray) or not isinstance(pq_map, dict):
        raise RuntimeError("hw_state must include lut_memory and posting PQ codes")
    opq = hw.get("opq_rotation")
    q_post = _query_post_matrix(raw, opq if isinstance(opq, np.ndarray) else None)
    pq_all = _reconstruct_pq_all(lut, pq_map, max_rows=max_pq_rows)
    if pq_all.shape[0] < 2 or q_post.shape[0] < 1:
        raise RuntimeError("Not enough PQ-recon or query rows for calibration")

    n_q = min(int(n_queries), int(q_post.shape[0]))
    n_c = min(int(n_candidates), int(pq_all.shape[0]))
    rng = np.random.default_rng(int(seed))
    q_ix = rng.choice(q_post.shape[0], size=n_q, replace=q_post.shape[0] < n_q)
    c_ix = rng.choice(pq_all.shape[0], size=n_c, replace=pq_all.shape[0] < n_c)
    Q = np.asarray(q_post[q_ix], dtype=np.float64)
    C = np.asarray(pq_all[c_ix], dtype=np.float64)

    if device is None:
        tf_dir = tempfile.mkdtemp(prefix="aimdc_k_cal_")
        scaler_tmp = os.path.join(tf_dir, "scaler.json")
        try:
            device = _fit_device_on_pq_recon(
                device_fit_path,
                pq_all.astype(np.float32),
                scaler_path=scaler_tmp,
                device_reduce=device_reduce,
                device_use_log=device_use_log,
                v_min=v_min,
                v_max=v_max,
            )
        finally:
            try:
                if os.path.isfile(scaler_tmp):
                    os.remove(scaler_tmp)
                if os.path.isdir(tf_dir):
                    os.rmdir(tf_dir)
            except OSError:
                pass

    ratios: List[float] = []
    for qi in range(Q.shape[0]):
        q = Q[qi]
        dq_sw = np.linalg.norm(C - q[None, :], axis=1)
        dq_dev = np.asarray(device.pairwise(q.astype(np.float64), C.astype(np.float64)), dtype=np.float64)
        med_sw = float(np.median(np.abs(dq_sw)))
        med_dev = float(np.median(np.abs(dq_dev)))
        if med_sw < 1e-12 or med_dev < 1e-12:
            continue
        ratios.append(med_sw / med_dev)
    if not ratios:
        raise RuntimeError("All queries skipped (zero median distances); increase sample sizes.")
    arr = np.asarray(ratios, dtype=np.float64)
    k = float(np.median(arr))
    p5, p50, p95 = (float(np.percentile(arr, p)) for p in (5.0, 50.0, 95.0))
    payload: Dict[str, Any] = {
        "k": k,
        "n_queries": int(len(ratios)),
        "n_candidates": int(n_c),
        "ratio_stats": {"p5": p5, "p50": p50, "p95": p95},
        "seed": int(seed),
        "device_fit_path": os.path.basename(os.path.abspath(device_fit_path)),
        "scaler_source": "PQ-recon",
        "schema_version": 2,
        "reduction_mode": "mean" if device_reduce == "mean" else "sum",
        "fit_sha256": _fit_sha256(device_fit_path),
    }
    # Record the operating window so the runtime can detect a window mismatch.
    # (device may be a custom object without v_min/v_max, e.g. in unit tests.)
    dv_min = getattr(device, "v_min", None)
    dv_max = getattr(device, "v_max", None)
    if dv_min is not None and dv_max is not None:
        payload["v_min"] = float(dv_min)
        payload["v_max"] = float(dv_max)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate AIMDC scale_k vs software L2.")
    p.add_argument("--offline_dir", required=True)
    p.add_argument("--device_fit_path", required=True)
    p.add_argument("--n_queries", type=int, default=128)
    p.add_argument("--n_candidates", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--device_reduce", type=str, default="sum", choices=["sum", "mean"])
    p.add_argument("--device_use_log", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v_min", type=float, default=-3.0,
                   help="Operating voltage window lower bound (default -3.0, matches MMR device path).")
    p.add_argument("--v_max", type=float, default=3.0,
                   help="Operating voltage window upper bound (default 3.0).")
    args = p.parse_args()
    payload = run_calibration(
        args.offline_dir,
        args.device_fit_path,
        n_queries=int(args.n_queries),
        n_candidates=int(args.n_candidates),
        seed=int(args.seed),
        device_reduce=str(args.device_reduce),
        device_use_log=bool(args.device_use_log),
        v_min=float(args.v_min),
        v_max=float(args.v_max),
    )
    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    rs = payload["ratio_stats"]
    print(
        f"[calibrate_k] n_queries={payload['n_queries']} (kept), "
        f"n_candidates={payload['n_candidates']}"
    )
    print(
        f"[calibrate_k] ratio percentiles: p5={rs['p5']:.4e}  p50={rs['p50']:.4e}  p95={rs['p95']:.4e}"
    )
    print(f"[calibrate_k] k (median of per-query medians) = {payload['k']:.4e}")
    if "v_min" in payload and "v_max" in payload:
        print(
            f"[calibrate_k] operating window: "
            f"v_min={payload['v_min']:g}, v_max={payload['v_max']:g}"
        )
    print(f"[calibrate_k] Saved: {out_path}")
    print(f"[calibrate_k] Apply at runtime: --device_scale_k_path {out_path}")


if __name__ == "__main__":
    main()
