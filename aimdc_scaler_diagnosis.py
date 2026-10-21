"""
AIMDC scaler distribution helpers (offline PQ vs raw/post-OPQ).
Used by scripts/diagnose_aimdc_scaler.py, tests, and online_accelerator_device warning.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np

PercentileDict = Dict[str, float]
MomentsDict = Dict[str, float]


def _pct(x: np.ndarray, ps: Sequence[float]) -> PercentileDict:
    flat = np.asarray(x, dtype=np.float64).ravel()
    if flat.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    out: PercentileDict = {}
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(flat, p))
    return out


def _moments(x: np.ndarray) -> MomentsDict:
    flat = np.asarray(x, dtype=np.float64).ravel()
    if flat.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "abs_max": float("nan")}
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat, ddof=0)),
        "abs_max": float(np.max(np.abs(flat))),
    }


def reconstruct_pq_embeddings(
    lut_memory: np.ndarray,
    pq_codes_matrix: np.ndarray,
) -> np.ndarray:
    """(N, M) uint/int codes -> (N, M*sub_dim) float32 rows."""
    lut = np.asarray(lut_memory, dtype=np.float32)
    codes = np.asarray(pq_codes_matrix, dtype=np.int64)
    if lut.ndim != 3:
        raise ValueError("lut_memory must be (M, K, sub_dim)")
    m, k_size, sub_dim = lut.shape
    if codes.ndim != 2 or codes.shape[1] != m:
        raise ValueError(f"pq_codes_matrix must be (N, {m}), got {codes.shape}")
    n = codes.shape[0]
    c_clipped = np.clip(codes, 0, k_size - 1)
    out = np.empty((n, m * sub_dim), dtype=np.float32)
    for sub in range(m):
        out[:, sub * sub_dim : (sub + 1) * sub_dim] = lut[sub, c_clipped[:, sub], :]
    return out


def gather_pq_codes_matrix(
    pq_codes: Mapping[Any, Any],
    lut_m: int,
    *,
    max_rows: int,
) -> np.ndarray:
    """First ``max_rows`` valid pq_code rows from posting lists (deterministic order)."""
    rows_buf: List[np.ndarray] = []
    if not isinstance(pq_codes, dict):
        return np.zeros((0, lut_m), dtype=np.int64)
    for _cid, posting_list in pq_codes.items():
        if len(rows_buf) >= max_rows:
            break
        if not isinstance(posting_list, list):
            continue
        for entry in posting_list:
            if len(rows_buf) >= max_rows:
                break
            if not isinstance(entry, dict):
                continue
            pc = entry.get("pq_code")
            if pc is None:
                continue
            v = np.asarray(pc, dtype=np.int64).ravel()
            if v.size != lut_m:
                continue
            rows_buf.append(v.copy())
    if not rows_buf:
        return np.zeros((0, lut_m), dtype=np.int64)
    return np.stack(rows_buf, axis=0).astype(np.int64)


def recommendation_from_delta(
    p1_rel: float,
    p99_rel: float,
    std_ratio: float,
) -> str:
    mx = max(p1_rel, p99_rel)
    if mx < 0.10 and 0.85 <= std_ratio <= 1.15:
        return "single_scaler"
    if mx > 0.20 or std_ratio < 0.7 or std_ratio > 1.4:
        return "dual_scaler"
    return "single_scaler_with_union_pool"


def build_summary_payload(
    *,
    q_pre: np.ndarray,
    q_post: np.ndarray,
    pq_recon: np.ndarray,
    has_opq_rotation: bool,
    ps: Tuple[float, ...] = (1.0, 5.0, 50.0, 95.0, 99.0),
) -> Dict[str, Any]:
    pct_pre = _pct(q_pre, ps)
    pct_post = _pct(q_post, ps)
    pct_pq = _pct(pq_recon, ps)
    mom_post = _moments(q_post)
    mom_pq = _moments(pq_recon)
    post_p1 = float(pct_post.get("p1", 0.0))
    post_p99 = float(pct_post.get("p99", 0.0))
    pq_p1 = float(pct_pq.get("p1", 0.0))
    pq_p99 = float(pct_pq.get("p99", 0.0))
    post_std = mom_post["std"] + 1e-12
    pq_std = mom_pq["std"] + 1e-12
    d1 = abs(pq_p1 - post_p1) / (abs(post_p1) + 1e-12)
    d99 = abs(pq_p99 - post_p99) / (abs(post_p99) + 1e-12)
    std_ratio = pq_std / post_std
    rec = recommendation_from_delta(d1, d99, std_ratio)
    return {
        "n_raw": int(q_post.shape[0]),
        "n_pq_recon": int(pq_recon.shape[0]),
        "dim": int(q_post.shape[1]) if q_post.ndim == 2 else 0,
        "has_opq_rotation": bool(has_opq_rotation),
        "percentiles": {
            "raw_pre_opq": pct_pre,
            "raw_post_opq": pct_post,
            "pq_recon": pct_pq,
        },
        "moments": {
            "raw_post_opq": mom_post,
            "pq_recon": mom_pq,
        },
        "delta_vs_post_opq": {
            "p1_rel": float(d1),
            "p99_rel": float(d99),
            "std_ratio": float(std_ratio),
        },
        "recommendation": rec,
    }


def affine_voltage(
    x: np.ndarray,
    x_lo: float,
    x_hi: float,
    v_min: float = -1.0,
    v_max: float = 1.0,
) -> np.ndarray:
    denom = x_hi - x_lo
    if denom <= 0.0:
        denom = 1e-12
    t = (np.asarray(x, dtype=np.float64) - x_lo) / denom
    return v_min + (v_max - v_min) * t


def clip_fraction(v: np.ndarray, v_min: float = -1.0, v_max: float = 1.0) -> float:
    vv = np.asarray(v, dtype=np.float64).ravel()
    if vv.size == 0:
        return 0.0
    return float(np.mean((vv < v_min) | (vv > v_max)))


def reconstruct_pq_sample_from_hw(
    hw_state: Mapping[str, Any],
    n: int = 2000,
) -> np.ndarray:
    """
    Up to n PQ-reconstructed rows (float32), same convention as step4_candidate_load.
    """
    lut = hw_state.get("lut_memory")
    pq_codes = hw_state.get("l2_accelerator_2_pq_codes")
    if lut is None or not isinstance(lut, np.ndarray) or lut.ndim != 3:
        raise ValueError("hw_state['lut_memory'] must be a (M,K,sub_dim) ndarray")
    m, _k, _sd = lut.shape
    mat = gather_pq_codes_matrix(
        cast(Mapping[Any, Any], pq_codes or {}),
        m,
        max_rows=max(1, int(n)),
    )
    if mat.size == 0:
        return np.zeros((0, m * _sd), dtype=np.float32)
    return reconstruct_pq_embeddings(lut, mat)


def compare_p99_raw_vs_pq(
    emb_raw: np.ndarray,
    pq_recon: np.ndarray,
) -> float:
    """Relative |Δp99| for startup warning."""
    r = np.asarray(emb_raw, dtype=np.float64).ravel()
    p = np.asarray(pq_recon, dtype=np.float64).ravel()
    if r.size == 0 or p.size == 0:
        return 0.0
    raw_p99 = float(np.percentile(r, 99))
    pq_p99 = float(np.percentile(p, 99))
    return abs(pq_p99 - raw_p99) / (abs(raw_p99) + 1e-12)


def load_offline_scaler_inputs(offline_dir: str) -> Dict[str, Any]:
    """
    Load raw_embeddings, opq_rotation, lut_memory, pq posting dict from offline_data_*.
    Raises FileNotFoundError if raw_embeddings or pq_codebooks/posting_lists missing.
    """
    d = os.path.abspath(offline_dir)
    path_raw = os.path.join(d, "raw_embeddings.npy")
    path_opq = os.path.join(d, "opq_rotation.npy")
    path_lut_cb = os.path.join(d, "pq_codebooks.npy")
    path_lut_alt = os.path.join(d, "lut_memory.npy")
    path_pl_main = os.path.join(d, "posting_lists.pkl")
    path_pl_alt = os.path.join(d, "pq_codes.pkl")
    if not os.path.isfile(path_raw):
        raise FileNotFoundError(f"Missing raw_embeddings.npy under {d}")
    raw = np.load(path_raw)
    opq = np.load(path_opq) if os.path.isfile(path_opq) else None
    if os.path.isfile(path_lut_cb):
        path_lut = path_lut_cb
    elif os.path.isfile(path_lut_alt):
        path_lut = path_lut_alt
    else:
        raise FileNotFoundError(
            f"Missing PQ codebook array under {d}: expected pq_codebooks.npy or lut_memory.npy "
            "(same tensor as hw_state['lut_memory'])."
        )
    lut = np.load(path_lut)
    if os.path.isfile(path_pl_main):
        path_pl = path_pl_main
    elif os.path.isfile(path_pl_alt):
        path_pl = path_pl_alt
    else:
        raise FileNotFoundError(
            f"Missing posting PQ-code pickle under {d}: expected posting_lists.pkl or pq_codes.pkl."
        )
    with open(path_pl, "rb") as f:
        posting_lists = pickle.load(f)
    return {
        "raw_embeddings": raw,
        "opq_rotation": opq,
        "lut_memory": lut,
        "l2_accelerator_2_pq_codes": posting_lists,
    }


def run_diagnosis(
    offline_dir: str,
    out_dir: str,
    *,
    sample_size: Optional[int] = 20000,
    v_min: float = -1.0,
    v_max: float = 1.0,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Load offline, compute stats, write summary + PNGs. Returns summary dict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_offline_scaler_inputs(offline_dir)
    raw = np.asarray(data["raw_embeddings"], dtype=np.float32)
    opq = data["opq_rotation"]
    lut = np.asarray(data["lut_memory"], dtype=np.float32)
    pq_dict = cast(Mapping[Any, Any], data["l2_accelerator_2_pq_codes"] or {})
    rot_arr = opq if isinstance(opq, np.ndarray) else None
    n_raw, dim = raw.shape
    has_rot = (
        rot_arr is not None
        and rot_arr.size > 0
        and rot_arr.ndim == 2
        and rot_arr.shape[0] == dim
    )
    if has_rot:
        rot = np.asarray(rot_arr, dtype=np.float32)
        q_post = (raw @ rot).astype(np.float32)
    else:
        q_post = raw.copy()
    q_pre = raw.copy()
    m, _k, sub_dim = lut.shape
    mem_budget_rows = max(
        1000,
        min(
            int((2 * raw.nbytes) / max(dim * 4, 1)),
            5_000_000,
        ),
    )
    pq_mat = gather_pq_codes_matrix(pq_dict, m, max_rows=mem_budget_rows)
    if pq_mat.size == 0:
        raise RuntimeError("No valid pq_code rows found in posting_lists")
    pq_recon = reconstruct_pq_embeddings(lut, pq_mat)
    rng_q = np.random.default_rng(int(rng_seed))
    rng_p = np.random.default_rng(int(rng_seed) + 1)
    ss = int(sample_size) if sample_size is not None else 0
    if ss > 0 and q_post.shape[0] > ss:
        idx = rng_q.choice(q_post.shape[0], size=ss, replace=False)
        q_post_s = q_post[idx]
        q_pre_s = q_pre[idx]
    else:
        q_post_s = q_post
        q_pre_s = q_pre
    if ss > 0 and pq_recon.shape[0] > ss:
        idxp = rng_p.choice(pq_recon.shape[0], size=ss, replace=False)
        pq_recon_s = pq_recon[idxp]
    else:
        pq_recon_s = pq_recon

    summary = build_summary_payload(
        q_pre=q_pre_s,
        q_post=q_post_s,
        pq_recon=pq_recon_s,
        has_opq_rotation=bool(has_rot),
    )
    rec = str(summary["recommendation"])

    def clip_frac(x: np.ndarray, x_lo: float, x_hi: float) -> float:
        v = affine_voltage(x, x_lo, x_hi, v_min, v_max)
        return clip_fraction(v, v_min, v_max)

    union = np.concatenate([q_post_s.ravel(), pq_recon_s.ravel()])
    x_lo_u = float(np.percentile(union, 1.0))
    x_hi_u = float(np.percentile(union, 99.0))
    x_lo_q = float(np.percentile(q_post_s.ravel(), 1.0))
    x_hi_q = float(np.percentile(q_post_s.ravel(), 99.0))
    x_lo_p = float(np.percentile(pq_recon_s.ravel(), 1.0))
    x_hi_p = float(np.percentile(pq_recon_s.ravel(), 99.0))
    clip_q_u = clip_frac(q_post_s, x_lo_u, x_hi_u)
    clip_p_u = clip_frac(pq_recon_s, x_lo_u, x_hi_u)
    summary["clip_union_scaler"] = {
        "x_lo": x_lo_u,
        "x_hi": x_hi_u,
        "clip_fraction_q_post": clip_q_u,
        "clip_fraction_pq_recon": clip_p_u,
    }
    summary["n_raw_full"] = int(n_raw)
    summary["n_pq_rows_gathered"] = int(pq_recon.shape[0])

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    # B) percentiles.png
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    qv = q_post_s.ravel()
    pv = pq_recon_s.ravel()
    lo_q, hi_q = float(np.percentile(qv, 1)), float(np.percentile(qv, 99))
    lo_p, hi_p = float(np.percentile(pv, 1)), float(np.percentile(pv, 99))
    span_q = max(hi_q - lo_q, 1e-12)
    span_p = max(hi_p - lo_p, 1e-12)
    use_symlog = max(span_q, span_p) / min(span_q, span_p) > 10.0
    x_lo = min(lo_q, lo_p)
    x_hi = max(hi_q, hi_p)
    pad = max((x_hi - x_lo) * 0.05, 1e-6)
    linthresh = max(1e-4, min(span_q, span_p) * 0.05)
    for ax, arr, title, color in (
        (axes[0], qv, "query post-OPQ (or raw if no rot)", "steelblue"),
        (axes[1], pv, "PQ reconstruction", "darkorange"),
    ):
        if use_symlog:
            ax.hist(arr, bins=80, color=color, alpha=0.75, density=True)
            ax.set_xscale("symlog", linthresh=linthresh)
        else:
            ax.hist(
                arr,
                bins=80,
                range=(x_lo - pad, x_hi + pad),
                color=color,
                alpha=0.75,
                density=True,
            )
        p1, p99 = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
        ax.axvline(p1, color="k", ls="--", lw=1)
        ax.axvline(p99, color="k", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("value")
        if not use_symlog:
            ax.set_xlim(x_lo - pad, x_hi + pad)
    axes[0].set_ylabel("density")
    fig.suptitle("Global value distribution (p1 / p99 vertical lines)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "percentiles.png"), dpi=160)
    plt.close(fig)

    # C) per_dim_percentiles
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    d_i = np.arange(dim, dtype=int)
    p1q = np.percentile(q_post_s, 1, axis=0)
    p99q = np.percentile(q_post_s, 99, axis=0)
    p1p = np.percentile(pq_recon_s, 1, axis=0)
    p99p = np.percentile(pq_recon_s, 99, axis=0)
    axes[0].plot(d_i, p1q, label="post-OPQ / raw query", color="steelblue")
    axes[0].plot(d_i, p1p, label="PQ-recon", color="darkorange")
    axes[0].set_ylabel("p1")
    axes[0].legend()
    axes[0].set_title("Per-dimension p1")
    axes[1].plot(d_i, p99q, label="post-OPQ / raw query", color="steelblue")
    axes[1].plot(d_i, p99p, label="PQ-recon", color="darkorange")
    axes[1].set_ylabel("p99")
    axes[1].set_xlabel("dimension")
    axes[1].set_title("Per-dimension p99")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_dim_percentiles.png"), dpi=160)
    plt.close(fig)

    # D) voltage_preview
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    if rec == "dual_scaler":
        vq = affine_voltage(q_post_s, x_lo_q, x_hi_q, v_min, v_max)
        vp = affine_voltage(pq_recon_s, x_lo_p, x_hi_p, v_min, v_max)
        title = (
            f"voltage_preview (dual)  clip_q={clip_fraction(vq):.3%}  clip_p={clip_fraction(vp):.3%}"
        )
    else:
        vq = affine_voltage(q_post_s, x_lo_u, x_hi_u, v_min, v_max)
        vp = affine_voltage(pq_recon_s, x_lo_u, x_hi_u, v_min, v_max)
        title = (
            f"voltage_preview (union scaler)  clip_q={clip_fraction(vq):.3%}  "
            f"clip_p={clip_fraction(vp):.3%}"
        )
    for ax, arr, lab, c in (
        (axes[0], vq.ravel(), "V from q_post (V_SL side)", "steelblue"),
        (axes[1], vp.ravel(), "V from pq_recon (V_PRG side)", "darkorange"),
    ):
        ax.hist(arr, bins=80, color=c, alpha=0.75, density=True, range=(v_min - 0.5, v_max + 0.5))
        ax.axvline(v_min, color="k", lw=1)
        ax.axvline(v_max, color="k", lw=1)
        ax.set_title(lab)
        ax.set_xlabel("V")
    axes[0].set_ylabel("density")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "voltage_preview.png"), dpi=160)
    plt.close(fig)

    return summary


def print_diagnosis_stdout(summary: Dict[str, Any], *, out_dir: str) -> None:
    """Human-readable block after ``run_diagnosis`` (matches acceptance template)."""
    n_raw = int(summary["n_raw"])
    n_pq = int(summary["n_pq_recon"])
    dim = int(summary["dim"])
    has_rot = bool(summary["has_opq_rotation"])
    if has_rot:
        space = "ambiguous (see summary.json: raw_pre_opq vs raw_post_opq; query uses post-OPQ)"
    else:
        space = "post-OPQ (no opq_rotation.npy; stored rows used as query stream)"
    post = summary["percentiles"]["raw_post_opq"]
    pq = summary["percentiles"]["pq_recon"]
    mom_q = summary["moments"]["raw_post_opq"]
    mom_p = summary["moments"]["pq_recon"]
    d = summary["delta_vs_post_opq"]
    clip = summary.get("clip_union_scaler", {})
    rec = str(summary["recommendation"])
    p1q, p99q = post["p1"], post["p99"]
    p1p, p99p = pq["p1"], pq["p99"]
    std_q = mom_q["std"]
    std_p = mom_p["std"]
    cq = float(clip.get("clip_fraction_q_post", 0.0))
    cp = float(clip.get("clip_fraction_pq_recon", 0.0))
    print("=== AIMDC scaler diagnosis ===")
    print(f"raw_embeddings: shape=({n_raw}, {dim}), space={space}")
    print(f"PQ-recon:       shape=({n_pq}, {dim})")
    print("Percentile bounds (p1, p99):")
    print(f"  query post-OPQ:  [{p1q:+.4f}, {p99q:+.4f}]   std={std_q:.4f}")
    print(f"  PQ-recon:        [{p1p:+.4f}, {p99p:+.4f}]   std={std_p:.4f}")
    print(
        f"  Delta_p1_rel = {d['p1_rel'] * 100:.1f}%   "
        f"Delta_p99_rel = {d['p99_rel'] * 100:.1f}%   "
        f"std_ratio = {d['std_ratio']:.2f}"
    )
    print("Clip fraction with single scaler (fit on union):")
    print(f"  q_post  -> V: outside-[-1,1] = {cq * 100:.1f}%")
    print(f"  pq_recon -> V: outside-[-1,1] = {cp * 100:.1f}%")
    print(f"Recommendation: {rec}")
    if rec == "dual_scaler":
        print("  → Proceed to separate fit_scaler_vsl / fit_scaler_vprg (next patch).")
    elif rec == "single_scaler_with_union_pool":
        print("  → Fit one scaler on vstack([q_post, pq_recon]) percentiles.")
    od = os.path.abspath(out_dir)
    names = ("summary.json", "percentiles.png", "per_dim_percentiles.png", "voltage_preview.png")
    print("Artifacts: " + ", ".join(os.path.join(od, n) for n in names))
