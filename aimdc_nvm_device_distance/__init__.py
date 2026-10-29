"""
Analytical NVM (FeFET) bit-cell distance for centroid search.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping, Optional, cast

import numpy as np
from numpy.typing import NDArray

from aimdc_variation import FrozenCellVariation

from h_aimdc_sim.aimdc_sim_core import DEFAULT_CIRCUIT_PARAMS, load_fit_json

K_B = 1.380649e-23
SUB_CAP = 5.0

# linear_sum: hardware match-line (sum currents, then optional p-root).
# legacy_*: old log-domain reduce; emit DeprecationWarning when selected.
ReduceMode = Literal["linear_sum", "legacy_sum", "legacy_mean", "sum", "mean"]
_LEGACY_REDUCE = frozenset({"sum", "mean", "legacy_sum", "legacy_mean"})
_FEFET_MODEL_PREFIXES = ("fefet_ref_",)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_reduce(reduce: str, *, is_fefet: bool) -> str:
    r = str(reduce)
    if r in ("sum", "mean"):
        warnings.warn(
            f"NvmDeviceDistance: reduce={r!r} is deprecated; use "
            f"'legacy_{r}' or 'linear_sum'. "
            "Log-domain sum/mean do not match hardware ML current summation.",
            DeprecationWarning,
            stacklevel=3,
        )
        return f"legacy_{r}"
    if r in ("legacy_sum", "legacy_mean"):
        warnings.warn(
            f"NvmDeviceDistance: reduce={r!r} is a legacy log-domain mode; "
            "prefer reduce='linear_sum' for FeFET hardware fidelity.",
            DeprecationWarning,
            stacklevel=3,
        )
        return r
    if r == "linear_sum":
        return r
    return "linear_sum" if is_fefet else "legacy_sum"


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
    # Device-to-device variation: when True, sigma_alpha, sigma_log_mu and sigma_vth are
    # drawn once per cell (see aimdc_variation.py) instead of on every read. Alpha and
    # gain are common to the two branches of a cell; threshold offsets are per branch.
    frozen_variation: bool = False
    variation_seed: Optional[int] = None

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
        if k in ("shot_enable", "frozen_variation"):
            kwargs[k] = bool(v)
        elif k == "variation_seed":
            kwargs[k] = None if v is None else int(v)
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


def _igzo_log10_v3(
    v_gs: NDArray[np.float64],
    theta,  # iterable of 8 broadcastable arrays or scalars
) -> NDArray[np.float64]:
    # theta[2] = ss_v_per_dec (subthreshold swing); not the VSS rail.
    I_off, V_th, ss_v_per_dec, I0_sub, I_sat_max, V_sat_scale, gamma, delta = (
        theta[0],
        theta[1],
        theta[2],
        theta[3],
        theta[4],
        theta[5],
        theta[6],
        theta[7],
    )
    v_gst = v_gs - V_th
    v_gst_smooth = 0.5 * (v_gst + np.sqrt(v_gst * v_gst + delta * delta))
    sub_arg = SUB_CAP * np.tanh(v_gst / ss_v_per_dec / SUB_CAP)
    i_sub = I0_sub * np.exp(sub_arg)
    tanh_part = np.tanh(v_gst_smooth / V_sat_scale)
    from h_aimdc_sim.igzo_powerlaw_model import POWERLAW_EPS

    i_above = I_sat_max * np.power(np.maximum(tanh_part, 0.0) + POWERLAW_EPS, gamma)
    ids = I_off + i_sub + i_above
    return cast(
        NDArray[np.float64],
        np.log10(np.maximum(ids, 1e-30)).astype(np.float64, copy=False),
    )


class NvmDeviceDistance:
    """Embedding -> voltage -> per-dim I_bit -> reduced scalar distance."""

    def __init__(
        self,
        fit_json_path: str,
        scaler_path: Optional[str] = None,
        *,
        reduce: Optional[ReduceMode] = None,
        use_log: bool = True,
        eps: float = 1e-15,
        v_ml: float = 0.0,
        noise: Optional[NoiseModel] = None,
        inverter_vtc_path: Optional[str] = None,
        p_root: Optional[float] = None,
        v_fold: float = 0.0,
    ) -> None:
        self.fit_json_path = os.path.abspath(fit_json_path)
        self.fit_sha256 = _sha256_file(self.fit_json_path)
        popt, meta = load_fit_json(self.fit_json_path)
        self._meta: dict[str, Any] = dict(meta)
        model_name = str(self._meta.get("model", "") or "")
        self._is_fefet = bool(
            any(model_name.startswith(p) for p in _FEFET_MODEL_PREFIXES)
            or self._meta.get("reduction_mode") == "linear_sum"
            or (
                isinstance(self._meta.get("backend"), str)
                and self._meta.get("backend") in ("lut", "powerlaw3")
            )
        )
        # Legacy IGZO path still expects an 8-vector; FeFET LUT may have empty popt.
        popt_a = np.asarray(popt, dtype=np.float64).ravel()
        self._fefet_model = None
        if self._is_fefet:
            backend = self._meta.get("backend")
            has_lut = backend == "lut" and "v_grid" in self._meta
            # load_fit_json strips arrays into meta only if present in JSON;
            # re-read via load_fefet_ref_fit when the file is a real FeFET fit.
            try:
                from h_aimdc_sim.fefet_ref_model import load_fefet_ref_fit

                self._fefet_model = load_fefet_ref_fit(self.fit_json_path)
            except (KeyError, ValueError, TypeError):
                self._fefet_model = None
            self._theta = (
                np.zeros(0, dtype=np.float64)
                if popt_a.size == 0
                else popt_a.astype(np.float64, copy=False)
            )
        else:
            if popt_a.size != 8:
                raise ValueError(
                    f"NvmDeviceDistance: expected 8-param IGZO popt, got {popt_a.size} "
                    f"(model={model_name!r})."
                )
            self._theta = popt_a.reshape(8)
        cp = dict(DEFAULT_CIRCUIT_PARAMS)
        mcp = self._meta.get("circuit_params")
        if isinstance(mcp, dict):
            for kk, vv in mcp.items():
                if isinstance(vv, (int, float, np.integer, np.floating)):
                    cp[str(kk)] = float(vv)
        self._circuit_params: dict[str, float] = {
            str(k): float(v)
            for k, v in cp.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }
        self.v_min = float(self._circuit_params.get("v_min", -1.0))
        self.v_max = float(self._circuit_params.get("v_max", 1.0))
        self.v_fold = float(
            self._circuit_params.get("V_FOLD", v_fold if v_fold is not None else 0.0)
        )

        if reduce is None:
            reduce = "linear_sum" if self._is_fefet else "legacy_sum"
        self.reduce: str = _normalize_reduce(str(reduce), is_fefet=self._is_fefet)
        self.use_log = bool(use_log)
        if self.reduce == "linear_sum":
            # Hardware sums currents; log before reduce is incorrect.
            self.use_log = False
        self.eps = float(eps)
        self.v_ml = float(v_ml)
        # Minkowski-p from fit metadata (triode region ≈ 1.15); None → no root.
        meta_p = self._meta.get("power_law_p")
        self.p: Optional[float] = float(meta_p) if meta_p is not None else None
        if p_root is not None:
            self.p_root: Optional[float] = float(p_root)
        elif self.reduce == "linear_sum" and self.p is not None:
            self.p_root = None  # default: raw current sum; caller opts into root
        else:
            self.p_root = None

        self.noise: NoiseModel = noise if noise is not None else NoiseModel()
        self._rng = np.random.default_rng(self.noise.rng_seed)
        self._frozen: Optional[FrozenCellVariation] = (
            FrozenCellVariation(self.noise.variation_seed or 0)
            if self.noise.frozen_variation else None
        )
        self._inv_v_in: Optional[NDArray[np.float64]] = None
        self._inv_v_out: Optional[NDArray[np.float64]] = None
        if inverter_vtc_path:
            warnings.warn(
                "NvmDeviceDistance: inverter_vtc_path is deprecated with the "
                "2-FeFET cell (V_FOLD complement is exact). Ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.inverter_vtc_path: Optional[str] = None
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
                self._x_lo = None
                self._x_hi = None

    def reset_rng(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)

    def _require_scaler(self) -> None:
        if self._x_lo is None or self._x_hi is None:
            raise RuntimeError(
                "NvmDeviceDistance: scaler not set. Call fit_scaler() or load_scaler()."
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
            raise ValueError(
                f"NvmDeviceDistance: scaler fit_sha256 mismatch "
                f"(scaler={stored_sha}, fit={self.fit_sha256}). Re-fit the scaler."
            )
        self._x_lo = float(d["x_lo"])
        self._x_hi = float(d["x_hi"])
        self.v_min = float(d.get("v_min", self.v_min))
        self.v_max = float(d.get("v_max", self.v_max))
        p = d.get("pct", [1.0, 99.0])
        self._fit_pct = (float(p[0]), float(p[1]))
        self.embedding_dim = int(d.get("embedding_dim", 0))

    def _reduce(self, val: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reduce per-dim values. linear_sum always accumulates in float64."""
        v = np.asarray(val, dtype=np.float64)
        if self.reduce == "linear_sum":
            s = np.sum(v, axis=-1, dtype=np.float64)
            if self.p_root is not None and float(self.p_root) != 0.0:
                pr = float(self.p_root)
                # Guard: currents are ~1e-9..1e-6; dim~768 stays in float64.
                s = np.maximum(s, 0.0)
                with np.errstate(over="raise", invalid="raise"):
                    try:
                        s = np.power(s, 1.0 / pr)
                    except FloatingPointError as exc:
                        raise FloatingPointError(
                            "NvmDeviceDistance: overflow/underflow in p_root reduce"
                        ) from exc
            return cast(NDArray[np.float64], s)
        if self.reduce in ("legacy_sum", "sum"):
            return cast(NDArray[np.float64], np.sum(v, axis=-1, dtype=np.float64))
        return cast(NDArray[np.float64], np.mean(v, axis=-1, dtype=np.float64))

    def _inverter(self, v_a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Column-shared inverter transfer V_A -> V_B.

        If an inverter VTC LUT was provided at construction time, evaluate by
        1D linear interpolation (clamped to the LUT range via np.interp's default
        edge behavior). Otherwise fall back to the ideal sign-flip used in Phase 1.
        """
        if self._inv_v_in is None or self._inv_v_out is None:
            return -np.asarray(v_a, dtype=np.float64)
        v_a_arr = np.asarray(v_a, dtype=np.float64)
        return cast(
            NDArray[np.float64],
            np.interp(v_a_arr, self._inv_v_in, self._inv_v_out).astype(
                np.float64, copy=False
            ),
        )

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
        noise = self.noise
        rng = self._rng
        v_ml = self.v_ml

        if self._fefet_model is not None:
            # 2-FeFET cell: w = V_PRG, complement via V_FOLD.
            w = v_prg_b
            fz = self._frozen if noise.enabled() else None
            if fz is not None and noise.sigma_vth > 0.0:
                # stored-value (notch) offset of the cell, fixed for every read
                w = w + fz.field("vth", noise.sigma_vth, shape)
            elif noise.enabled() and noise.sigma_vth > 0.0:
                w = w + rng.normal(0.0, noise.sigma_vth, size=shape)
            if noise.enabled() and noise.retention_tau is not None and noise.t_hold > 0.0:
                w = w * np.exp(-noise.t_hold / float(noise.retention_tau))
            v_fold = float(self.v_fold)
            g1 = v_sl_b - w - v_ml
            g2 = (v_fold - v_sl_b) - (v_fold - w) - v_ml  # = w - v_sl - v_ml
            if fz is not None and noise.sigma_alpha > 0.0:
                # coupling (capacitor-divider) ratio of the cell scales its gate drive
                a = 1.0 + fz.field("alpha", noise.sigma_alpha, shape)
                g1 = a * (v_sl_b - w) - v_ml
                g2 = a * (w - v_sl_b) - v_ml
            # Evaluate ref model; allow_extrapolation for batch distance (scaler may
            # push edges). Validity is enforced at deck/config time.
            flat1 = g1.reshape(-1)
            flat2 = g2.reshape(-1)
            i1 = self._fefet_model.current(flat1, dv=0.0, allow_extrapolation=True)
            i2 = self._fefet_model.current(flat2, dv=0.0, allow_extrapolation=True)
            i_out = (i1 + i2).reshape(shape)
            if fz is not None and noise.sigma_log_mu > 0.0:
                # cell gain (log10 units), common to both FeFETs of the pair
                i_out = i_out * np.power(10.0, fz.field("gain", noise.sigma_log_mu, shape))
            if noise.enabled():
                if noise.flicker_sigma > 0.0:
                    i_out = i_out * (1.0 + rng.normal(0.0, noise.flicker_sigma, size=shape))
                if noise.rtn_sigma > 0.0:
                    i_out = i_out * (1.0 + rng.normal(0.0, noise.rtn_sigma, size=shape))
                if noise.shot_enable:
                    q_e = 1.602176634e-19
                    sigma_shot = np.sqrt(
                        2.0 * q_e * np.maximum(i_out, 0.0) * float(noise.read_bandwidth_hz)
                    )
                    i_out = np.maximum(i_out + rng.normal(0.0, sigma_shot, size=shape), 0.0)
                if noise.adc_bits is not None:
                    levels = 2 ** int(noise.adc_bits)
                    step = float(noise.adc_i_fullscale) / float(levels)
                    i_out = np.clip(
                        np.round(i_out / step) * step, 0.0, float(noise.adc_i_fullscale)
                    )
            return cast(NDArray[np.float64], i_out.astype(np.float64, copy=False))

        th = self._theta
        # Legacy IGZO + inverter path.
        v_b_inv = self._inverter(v_sl_b)
        if not noise.enabled():
            v_gs_a = v_sl_b - v_prg_b - v_ml
            v_gs_b = v_b_inv + v_prg_b - v_ml
            log_i_a = _igzo_log10_v3(v_gs_a, th)
            log_i_b = _igzo_log10_v3(v_gs_b, th)
            i_det = np.power(10.0, log_i_a) + np.power(10.0, log_i_b)
            return cast(NDArray[np.float64], i_det.astype(np.float64, copy=False))

        v_prg_eff = v_prg_b
        if noise.ktc_C > 0.0:
            sig = float(np.sqrt(K_B * noise.temperature / noise.ktc_C))
            v_prg_eff = v_prg_eff + rng.normal(0.0, sig, size=shape)
        if noise.retention_tau is not None and noise.t_hold > 0.0:
            v_prg_eff = v_prg_eff * np.exp(-noise.t_hold / float(noise.retention_tau))
        if noise.sigma_alpha > 0.0:
            alpha_a = 1.0 + rng.normal(0.0, noise.sigma_alpha, size=shape)
            alpha_b = 1.0 + rng.normal(0.0, noise.sigma_alpha, size=shape)
        else:
            alpha_a = np.ones(shape, dtype=np.float64)
            alpha_b = np.ones(shape, dtype=np.float64)

        v_gs_a = alpha_a * v_sl_b - v_prg_eff - v_ml
        v_gs_b = alpha_b * v_b_inv + v_prg_eff - v_ml

        th_a = [np.full(shape, float(th[i]), dtype=np.float64) for i in range(8)]
        th_b = [np.full(shape, float(th[i]), dtype=np.float64) for i in range(8)]

        if noise.sigma_vth > 0.0:
            th_a[1] = th_a[1] + rng.normal(0.0, noise.sigma_vth, size=shape)
            th_b[1] = th_b[1] + rng.normal(0.0, noise.sigma_vth, size=shape)
        if noise.sigma_log_mu > 0.0:
            s_a = rng.normal(0.0, noise.sigma_log_mu, size=shape)
            s_b = rng.normal(0.0, noise.sigma_log_mu, size=shape)
            th_a[0] = th_a[0] * np.power(10.0, s_a)
            th_a[4] = th_a[4] * np.power(10.0, s_a)
            th_b[0] = th_b[0] * np.power(10.0, s_b)
            th_b[4] = th_b[4] * np.power(10.0, s_b)

        log_i_a = _igzo_log10_v3(v_gs_a, th_a)
        log_i_b = _igzo_log10_v3(v_gs_b, th_b)
        i_out = np.power(10.0, log_i_a) + np.power(10.0, log_i_b)

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
        v_sl = aa[:, np.newaxis, :]
        v_prg = bb[np.newaxis, :, :]
        i_all = self._bit_current(v_sl, v_prg)
        if self.reduce == "linear_sum":
            # Hardware: I_ML = sum_d I_d  (linear domain, float64).
            val = np.asarray(i_all, dtype=np.float64)
        elif self.use_log:
            val = np.log10(i_all + self.eps)
        else:
            val = i_all
        out = self._reduce(val)
        return cast(NDArray[np.float64], np.asarray(out, dtype=np.float64))


__all__ = [
    "K_B",
    "NvmDeviceDistance",
    "NoiseModel",
    "noise_model_from_json_path",
    "noise_model_from_mapping",
]
