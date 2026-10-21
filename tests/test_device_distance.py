from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pytest
from scipy import stats  # type: ignore[import-untyped]

from aimdc_device_distance import DeviceDistance, NoiseModel, noise_model_from_mapping
from h_aimdc_sim.aimdc_sim_core import (
    composite_model,
    compute_fit_residual_rmse,
    fit_igzo_model,
    load_fit_json,
    save_fit_json,
)

_REPO = Path(__file__).resolve().parents[1]
CSV_PATH = _REPO / "h_aimdc_sim" / "igzo_measure.csv"


@pytest.fixture(scope="module")
def fitted() -> tuple[np.ndarray, Path]:
    assert CSV_PATH.is_file(), f"missing {CSV_PATH}"
    popt, _expr = fit_igzo_model(str(CSV_PATH))
    rmse, vds = compute_fit_residual_rmse(str(CSV_PATH), popt)
    out = _REPO / "h_aimdc_sim" / "igzo_fit.json"
    save_fit_json(
        popt,
        str(out),
        source_csv="igzo_measure.csv",
        vds_fit=vds,
        fit_rmse_log10=rmse,
    )
    return popt, out


def test_fit_roundtrip_rmse(fitted: tuple[np.ndarray, Path]) -> None:
    popt, _path = fitted
    rmse, _vds = compute_fit_residual_rmse(str(CSV_PATH), popt)
    assert rmse < 0.2


def test_vcurve_matches_formula(fitted: tuple[np.ndarray, Path]) -> None:
    _popt, fit_path = fitted
    dd = DeviceDistance(str(fit_path), scaler_path=None, noise=NoiseModel())
    v_sl = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    v_prg = np.zeros_like(v_sl)
    i_dd = np.asarray(dd.raw_bit_current(v_sl, v_prg), dtype=np.float64)
    th = dd._theta
    for i in range(len(v_sl)):
        dv = float(v_prg[i] - v_sl[i])
        log_l = composite_model(np.array([dv], dtype=np.float64), *th).item()
        log_r = composite_model(np.array([-dv], dtype=np.float64), *th).item()
        ref = 10.0**log_l + 10.0**log_r
        assert ref > 0
        rel = abs(i_dd[i] - ref) / ref
        assert rel < 0.01


def test_spearman_vs_l2(fitted: tuple[np.ndarray, Path]) -> None:
    _p, fit_path = fitted
    rng = np.random.default_rng(7)
    emb = rng.standard_normal((800, 768)).astype(np.float64)
    dd = DeviceDistance(str(fit_path), scaler_path=None, noise=NoiseModel())
    dd.fit_scaler(emb)
    d_dev: list[float] = []
    d_l2: list[float] = []
    for _ in range(800):
        i, j = int(rng.integers(0, 800)), int(rng.integers(0, 800))
        a, b = emb[i], emb[j]
        d_dev.append(dd.distance(a, b))
        d_l2.append(float(np.linalg.norm(a - b)))
    rho_bulk, _ = stats.spearmanr(d_dev, d_l2)
    assert float(rho_bulk) > 0.77
    u = rng.standard_normal(768)
    u = u / (float(np.linalg.norm(u)) + 1e-12)
    d_ray: list[float] = []
    t_ray: list[float] = []
    for t in np.linspace(0.05, 4.0, 120):
        d_ray.append(dd.distance(np.zeros_like(u), u * float(t)))
        t_ray.append(float(t))
    rho_ray, _ = stats.spearmanr(d_ray, t_ray)
    assert float(rho_ray) > 0.99


def test_symmetry_noise_off(fitted: tuple[np.ndarray, Path]) -> None:
    _p, fit_path = fitted
    rng = np.random.default_rng(1)
    dd = DeviceDistance(str(fit_path), scaler_path=None, noise=NoiseModel())
    dd.fit_scaler(rng.standard_normal((50, 32)))
    for _ in range(100):
        q = rng.standard_normal(32)
        c = rng.standard_normal(32)
        d1 = dd.distance(q, c)
        d2 = dd.distance(c, q)
        denom = 0.5 * (abs(d1) + abs(d2)) + 1e-12
        assert abs(d1 - d2) / denom < 1e-6


def test_self_distance_minimum(fitted: tuple[np.ndarray, Path]) -> None:
    _p, fit_path = fitted
    rng = np.random.default_rng(2)
    q = rng.standard_normal(32)
    others = rng.standard_normal((50, 32))
    stack = np.vstack([q.reshape(1, -1), others])
    dd = DeviceDistance(str(fit_path), scaler_path=None, noise=NoiseModel())
    dd.fit_scaler(stack)
    dvec = dd.pairwise(q, stack)
    assert int(np.argmin(dvec)) == 0


def test_determinism_under_seed(fitted: tuple[np.ndarray, Path]) -> None:
    _p, fit_path = fitted
    rng = np.random.default_rng(3)
    q = rng.standard_normal(16)
    c = rng.standard_normal(16)
    n = NoiseModel(sigma_vth=0.02, rng_seed=999)
    emb = rng.standard_normal((30, 16))
    vals: list[float] = []
    for _ in range(3):
        dd = DeviceDistance(str(fit_path), scaler_path=None, noise=n)
        dd.fit_scaler(emb)
        vals.append(dd.distance(q, c))
    assert abs(vals[0] - vals[1]) < 1e-15
    assert abs(vals[1] - vals[2]) < 1e-15


def test_noise_defaults_match_deterministic(fitted: tuple[np.ndarray, Path]) -> None:
    _p, fit_path = fitted
    rng = np.random.default_rng(4)
    q = rng.standard_normal(24)
    c = rng.standard_normal(24)
    d0 = DeviceDistance(str(fit_path), scaler_path=None, noise=NoiseModel())
    d1 = DeviceDistance(str(fit_path), scaler_path=None, noise=None)
    emb = rng.standard_normal((40, 24))
    d0.fit_scaler(emb)
    d1.fit_scaler(emb)
    x0, x1 = d0.distance(q, c), d1.distance(q, c)
    assert abs(x0 - x1) < 1e-12


def test_monte_carlo_sigma_vth_scaling(fitted: tuple[np.ndarray, Path]) -> None:
    _p, fit_path = fitted
    rng = np.random.default_rng(5)
    q = rng.standard_normal(24)
    c = rng.standard_normal(24)
    emb = rng.standard_normal((80, 24))
    sigmas = np.array([0.02, 0.04, 0.08, 0.16], dtype=np.float64)
    stds: list[float] = []
    for sig in sigmas:
        trials: list[float] = []
        for t in range(2048):
            dd = DeviceDistance(
                str(fit_path),
                scaler_path=None,
                noise=NoiseModel(sigma_vth=float(sig), rng_seed=1009 * t + 11),
            )
            dd.fit_scaler(emb)
            trials.append(dd.distance(q, c))
        stds.append(float(np.std(np.asarray(trials), ddof=0)))
    lx = np.log(sigmas)
    ly = np.log(np.maximum(np.asarray(stds, dtype=np.float64), 1e-20))
    slope, _intercept, _r, _p, _se = stats.linregress(lx, ly)
    assert abs(float(slope) - 1.0) < 0.35
    assert stds[-1] > stds[0] * 3.0


def test_scaler_persistence(fitted: tuple[np.ndarray, Path]) -> None:
    _p, fit_path = fitted
    rng = np.random.default_rng(6)
    emb = rng.standard_normal((100, 20))
    scaler_path = fit_path.parent / "test_tmp_scaler.json"
    try:
        dd = DeviceDistance(str(fit_path), scaler_path=None, noise=NoiseModel())
        dd.fit_scaler(emb)
        dd.save_scaler(str(scaler_path))
        q = rng.standard_normal(20)
        c = rng.standard_normal(20)
        d_a = dd.distance(q, c)
        dd2 = DeviceDistance(str(fit_path), scaler_path=str(scaler_path), noise=NoiseModel())
        d_b = dd2.distance(q, c)
        assert abs(d_a - d_b) < 1e-9
    finally:
        if scaler_path.is_file():
            scaler_path.unlink()


def test_noise_json_roundtrip() -> None:
    raw = {"sigma_vth": 0.01, "shot_enable": True, "adc_bits": None, "rng_seed": 3}
    n = noise_model_from_mapping(raw)
    assert n.sigma_vth == 0.01
    assert n.shot_enable is True


def test_load_fit_json_schema(fitted: tuple[np.ndarray, Path]) -> None:
    popt, path = fitted
    p2, meta = load_fit_json(str(path))
    assert np.allclose(popt, p2)
    assert meta.get("schema_version") == 1
    assert "circuit_params" in meta
