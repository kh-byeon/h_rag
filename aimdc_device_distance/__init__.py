"""
Analytical AIMDC bit-cell distance for MMR (no LUT, no Docker at inference).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping, Optional, cast

import numpy as np
from numpy.typing import NDArray

from h_aimdc_sim.aimdc_sim_core import DEFAULT_CIRCUIT_PARAMS, load_fit_json

K_B = 1.380649e-23

ReduceMode = Literal["sum", "mean"]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class NoiseModel:
    """All defaults off (deterministic). Toggle fields to enable mechanisms."""

    ktc_C: float = 0.0
    temperature: float = 300.0
    sigma_alpha: float = 0.0
    sigma_vth: float = 0.0
    sigma_log_mu: float = 0.0
    retention_tau: Optional[float] = None
    t_hold: float = 0.0
    flicker_sigma: float = 0.0
    shot_enable: bool = False
    read_bandwidth_hz: float = 1e6
    rtn_sigma: float = 0.0
    adc_bits: Optional[int] = None
    adc_i_fullscale: float = 1e-6
    rng_seed: Optional[int] = None

    def enabled(self) -> bool:
        return any(
            [
                self.ktc_C > 0.0,
                self.sigma_alpha > 0.0,
                self.sigma_vth > 0.0,
                self.sigma_log_mu > 0.0,
                self.retention_tau is not None,
                self.flicker_sigma > 0.0,
                self.shot_enable,
                self.rtn_sigma > 0.0,
                self.adc_bits is not None,
            ]
        )


def noise_model_from_mapping(data: Mapping[str, Any]) -> NoiseModel:
    """Build NoiseModel from JSON dict keys (unknown keys ignored)."""
    valid = {f.name for f in fields(NoiseModel)}
    kwargs: dict[str, Any] = {}
    for k, v in data.items():
        if k not in valid:
            continue
        if k == "shot_enable":
            kwargs[k] = bool(v)
        elif k == "adc_bits":
            kwargs[k] = None if v is None else int(v)
        elif k == "retention_tau":
            kwargs[k] = None if v is None else float(v)
        elif k == "rng_seed":
            kwargs[k] = None if v is None else int(v)
        else:
            kwargs[k] = float(v)
    return NoiseModel(**kwargs)


def noise_model_from_json_path(path: str) -> NoiseModel:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("noise JSON must be an object")
    return noise_model_from_mapping(cast(dict[str, Any], raw))


def _composite_log10(
    v: NDArray[np.float64],
    c0: NDArray[np.float64],
    k: NDArray[np.float64],
    vth: NDArray[np.float64],
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
) -> NDArray[np.float64]:
    s = 0.5 * (1.0 + np.tanh(k * (v - vth)))
    on_poly = p0 + p1 * v + p2 * v * v + p3 * v * v * v
    out = c0 * (1.0 - s) + on_poly * s
    return cast(NDArray[np.float64], out.astype(np.float64, copy=False))


class DeviceDistance:
    """Embedding → voltage → per-dim I_bit → reduced scalar distance."""

    def __init__(
        self,
        fit_json_path: str,
        scaler_path: Optional[str] = None,
        *,
        reduce: ReduceMode = "sum",
        use_log: bool = True,
        eps: float = 1e-15,
        v_ml: float = 0.0,
        noise: Optional[NoiseModel] = None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
    ) -> None:
        self.fit_json_path = os.path.abspath(fit_json_path)
        self.fit_sha256 = _sha256_file(self.fit_json_path)
        popt, meta = load_fit_json(self.fit_json_path)
        self._theta: NDArray[np.float64] = np.asarray(popt, dtype=np.float64).reshape(7)
        self._meta: dict[str, Any] = dict(meta)
        cp = dict(DEFAULT_CIRCUIT_PARAMS)
        mcp = self._meta.get("circuit_params")
        if isinstance(mcp, dict):
            for kk, vv in mcp.items():
                if isinstance(vv, (int, float, np.integer, np.floating)):
                    cp[str(kk)] = float(vv)
        self._circuit_params: dict[str, float] = {str(k): float(v) for k, v in cp.items()}
        self.v_min = float(self._circuit_params.get("v_min", -1.0))
        self.v_max = float(self._circuit_params.get("v_max", 1.0))
        self.reduce: ReduceMode = reduce if reduce in ("sum", "mean") else "sum"
        self.use_log = bool(use_log)
        self.eps = float(eps)
        self.v_ml = float(v_ml)
        self.noise: NoiseModel = noise if noise is not None else NoiseModel()
        self._rng = np.random.default_rng(self.noise.rng_seed)
        self._x_lo: Optional[float] = None
        self._x_hi: Optional[float] = None
        self._fit_pct: tuple[float, float] = (1.0, 99.0)
        self.embedding_dim: int = 0
        default_scaler = f"{os.path.splitext(self.fit_json_path)[0]}.scaler.json"
        self._scaler_path = (
            os.path.abspath(scaler_path) if scaler_path else default_scaler
        )
        if os.path.isfile(self._scaler_path):
            try:
                if os.path.getsize(self._scaler_path) > 0:
                    self.load_scaler(self._scaler_path)
            except (OSError, ValueError):
                # Missing/corrupt/stale (fit_sha256 mismatch) → leave unset for re-fit.
                self._x_lo = None
                self._x_hi = None
        # Explicit operating-window overrides win over both circuit_params and the
        # scaler's stored window (load_scaler above may have overwritten them).
        # Default None keeps the fit/scaler-derived window (byte-for-byte).
        if v_min is not None:
            self.v_min = float(v_min)
        if v_max is not None:
            self.v_max = float(v_max)

    def reset_rng(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)

    def _require_scaler(self) -> None:
        if self._x_lo is None or self._x_hi is None:
            raise RuntimeError(
                "DeviceDistance: scaler not set. Call fit_scaler() or load_scaler()."
            )

    def _scale(self, x: NDArray[Any]) -> NDArray[np.float64]:
        self._require_scaler()
        if self._x_lo is None or self._x_hi is None:
            raise RuntimeError("internal: scaler bounds unset after _require_scaler")
        x_lo = self._x_lo
        x_hi = self._x_hi
        denom = x_hi - x_lo
        if denom <= 0.0:
            denom = 1e-12
        t = (np.asarray(x, dtype=np.float64) - x_lo) / denom
        out = self.v_min + (self.v_max - self.v_min) * t
        return cast(NDArray[np.float64], out.astype(np.float64, copy=False))

    def fit_scaler(
        self,
        embeddings: NDArray[Any],
        pct: tuple[float, float] = (1.0, 99.0),
    ) -> None:
        flat = np.asarray(embeddings, dtype=np.float64).ravel()
        if flat.size == 0:
            raise ValueError("fit_scaler: empty embeddings")
        self._x_lo = float(np.percentile(flat, pct[0]))
        self._x_hi = float(np.percentile(flat, pct[1]))
        if self._x_hi <= self._x_lo:
            self._x_hi = self._x_lo + 1e-12
        self._fit_pct = (float(pct[0]), float(pct[1]))
        self.embedding_dim = int(np.asarray(embeddings).shape[-1])

    def save_scaler(self, path: str) -> None:
        self._require_scaler()
        payload = {
            "x_lo": self._x_lo,
            "x_hi": self._x_hi,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "pct": list(self._fit_pct),
            "embedding_dim": self.embedding_dim,
            "fit_sha256": self.fit_sha256,
            "reduction_mode": self.reduce,
        }
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_scaler(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        stored_sha = d.get("fit_sha256")
        if stored_sha is not None and str(stored_sha) != self.fit_sha256:
            # Stale scaler for a different fit — caller should re-fit.
            raise ValueError(
                f"DeviceDistance: scaler fit_sha256 mismatch "
                f"(scaler={stored_sha}, fit={self.fit_sha256}). Re-fit the scaler."
            )
        self._x_lo = float(d["x_lo"])
        self._x_hi = float(d["x_hi"])
        self.v_min = float(d.get("v_min", self.v_min))
        self.v_max = float(d.get("v_max", self.v_max))
        p = d.get("pct", [1.0, 99.0])
        self._fit_pct = (float(p[0]), float(p[1]))
        self.embedding_dim = int(d.get("embedding_dim", 0))
        self._scaler_fit_sha256 = stored_sha
        self._scaler_reduction_mode = d.get("reduction_mode")

    def _reduce(self, val: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.reduce == "sum":
            return cast(NDArray[np.float64], np.sum(val, axis=-1, dtype=np.float64))
        return cast(NDArray[np.float64], np.mean(val, axis=-1, dtype=np.float64))

    def _bit_current(
        self,
        v_sl: NDArray[Any],
        v_prg: NDArray[Any],
    ) -> NDArray[np.float64]:
        v_sl_a = np.asarray(v_sl, dtype=np.float64)
        v_prg_a = np.asarray(v_prg, dtype=np.float64)
        shape = tuple(np.broadcast_shapes(v_sl_a.shape, v_prg_a.shape))
        v_sl_b = np.broadcast_to(v_sl_a, shape)
        v_prg_b = np.broadcast_to(v_prg_a, shape)
        th = self._theta
        noise = self.noise
        rng = self._rng
        v_ml = self.v_ml

        if not noise.enabled():
            v_sn = v_prg_b - v_sl_b - v_ml
            v_sn_bar = -v_prg_b + v_sl_b - v_ml
            c0 = np.full(shape, float(th[0]), dtype=np.float64)
            k = np.full(shape, float(th[1]), dtype=np.float64)
            vth = np.full(shape, float(th[2]), dtype=np.float64)
            p0 = np.full(shape, float(th[3]), dtype=np.float64)
            p1 = np.full(shape, float(th[4]), dtype=np.float64)
            p2 = np.full(shape, float(th[5]), dtype=np.float64)
            p3 = np.full(shape, float(th[6]), dtype=np.float64)
            log_i_l = _composite_log10(v_sn, c0, k, vth, p0, p1, p2, p3)
            log_i_r = _composite_log10(v_sn_bar, c0, k, vth, p0, p1, p2, p3)
            i_det = np.power(10.0, log_i_l) + np.power(10.0, log_i_r)
            return cast(NDArray[np.float64], i_det.astype(np.float64, copy=False))

        v_prg_eff = v_prg_b
        if noise.ktc_C > 0.0:
            sig = float(np.sqrt(K_B * noise.temperature / noise.ktc_C))
            v_prg_eff = v_prg_eff + rng.normal(0.0, sig, size=shape)
        if noise.retention_tau is not None and noise.t_hold > 0.0:
            v_prg_eff = v_prg_eff * np.exp(-noise.t_hold / float(noise.retention_tau))
        if noise.sigma_alpha > 0.0:
            alpha_l = 1.0 + rng.normal(0.0, noise.sigma_alpha, size=shape)
            alpha_r = 1.0 + rng.normal(0.0, noise.sigma_alpha, size=shape)
        else:
            alpha_l = np.ones(shape, dtype=np.float64)
            alpha_r = np.ones(shape, dtype=np.float64)
        v_sn = v_prg_eff - alpha_l * v_sl_b - v_ml
        v_sn_bar = -v_prg_eff + alpha_r * v_sl_b - v_ml

        c0_l = np.full(shape, float(th[0]), dtype=np.float64)
        c0_r = np.full(shape, float(th[0]), dtype=np.float64)
        k_l = np.full(shape, float(th[1]), dtype=np.float64)
        k_r = np.full(shape, float(th[1]), dtype=np.float64)
        vth_l = np.full(shape, float(th[2]), dtype=np.float64)
        vth_r = np.full(shape, float(th[2]), dtype=np.float64)
        p0_l = np.full(shape, float(th[3]), dtype=np.float64)
        p0_r = np.full(shape, float(th[3]), dtype=np.float64)
        p1_a = np.full(shape, float(th[4]), dtype=np.float64)
        p2_a = np.full(shape, float(th[5]), dtype=np.float64)
        p3_a = np.full(shape, float(th[6]), dtype=np.float64)

        if noise.sigma_vth > 0.0:
            vth_l = vth_l + rng.normal(0.0, noise.sigma_vth, size=shape)
            vth_r = vth_r + rng.normal(0.0, noise.sigma_vth, size=shape)
        if noise.sigma_log_mu > 0.0:
            s_l = rng.normal(0.0, noise.sigma_log_mu, size=shape)
            s_r = rng.normal(0.0, noise.sigma_log_mu, size=shape)
            c0_l = c0_l + s_l
            p0_l = p0_l + s_l
            c0_r = c0_r + s_r
            p0_r = p0_r + s_r

        log_i_l = _composite_log10(v_sn, c0_l, k_l, vth_l, p0_l, p1_a, p2_a, p3_a)
        log_i_r = _composite_log10(v_sn_bar, c0_r, k_r, vth_r, p0_r, p1_a, p2_a, p3_a)
        i_out = np.power(10.0, log_i_l) + np.power(10.0, log_i_r)

        if noise.flicker_sigma > 0.0:
            i_out = i_out * (1.0 + rng.normal(0.0, noise.flicker_sigma, size=shape))
        if noise.rtn_sigma > 0.0:
            i_out = i_out * (1.0 + rng.normal(0.0, noise.rtn_sigma, size=shape))
        if noise.shot_enable:
            q_e = 1.602176634e-19
            sigma_shot = np.sqrt(
                2.0 * q_e * np.maximum(i_out, 0.0) * float(noise.read_bandwidth_hz)
            )
            i_out = i_out + rng.normal(0.0, sigma_shot, size=shape)
            i_out = np.maximum(i_out, 0.0)
        if noise.adc_bits is not None:
            levels = 2 ** int(noise.adc_bits)
            step = float(noise.adc_i_fullscale) / float(levels)
            i_out = np.clip(
                np.round(i_out / step) * step, 0.0, float(noise.adc_i_fullscale)
            )
        return cast(NDArray[np.float64], i_out.astype(np.float64, copy=False))

    def raw_bit_current(
        self,
        v_sl: NDArray[Any],
        v_prg: NDArray[Any],
    ) -> NDArray[np.float64]:
        """Physics I_bit(V_SL, V_PRG) without embedding scaler (noise hooks apply)."""
        return self._bit_current(v_sl, v_prg)

    def distance(self, q: NDArray[Any], c: NDArray[Any]) -> float:
        qv = np.asarray(q, dtype=np.float64).reshape(-1)
        cv = np.asarray(c, dtype=np.float64).reshape(1, -1)
        return float(self.pairwise(qv, cv)[0])

    def pairwise(self, q: NDArray[Any], c: NDArray[Any]) -> NDArray[np.float64]:
        qv = np.asarray(q, dtype=np.float64).reshape(1, -1)
        cm = np.asarray(c, dtype=np.float64)
        row = self.pair_matrix(qv, cm)[0]
        return cast(NDArray[np.float64], np.asarray(row, dtype=np.float64))

    def pair_matrix(self, a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.float64]:
        aa = self._scale(np.asarray(a, dtype=np.float64))
        bb = self._scale(np.asarray(b, dtype=np.float64))
        if aa.ndim != 2 or bb.ndim != 2:
            raise ValueError("pair_matrix expects 2D arrays (M,D) and (N,D)")
        if aa.shape[1] != bb.shape[1]:
            raise ValueError(f"D mismatch: {aa.shape[1]} vs {bb.shape[1]}")
        m, d1 = aa.shape
        n, d2 = bb.shape
        v_sl = aa[:, np.newaxis, :]
        v_prg = bb[np.newaxis, :, :]
        i_all = self._bit_current(v_sl, v_prg)
        if self.use_log:
            val = np.log10(i_all + self.eps)
        else:
            val = i_all
        out = self._reduce(val)
        return cast(NDArray[np.float64], np.asarray(out, dtype=np.float64))


__all__ = [
    "K_B",
    "DeviceDistance",
    "NoiseModel",
    "noise_model_from_json_path",
    "noise_model_from_mapping",
]
