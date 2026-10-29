"""Frozen (device-to-device) variation: fixed per cell, reproducible, and off by default."""
import numpy as np
import pytest

from aimdc_device_distance import DeviceDistance, NoiseModel, noise_model_from_mapping
from aimdc_variation import FrozenCellVariation

FIT = "h_aimdc_sim/igzo_fit.json"


def _unit(rng, shape):
    x = rng.normal(size=shape)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _dd(noise):
    dd = DeviceDistance(FIT, v_min=-3.0, v_max=3.0, noise=noise)
    return dd


def test_mapping_parses_new_keys():
    nm = noise_model_from_mapping({"frozen_variation": True, "variation_seed": 3, "sigma_alpha": 0.05})
    assert nm.frozen_variation is True and nm.variation_seed == 3 and nm.enabled()


def test_frozen_is_reproducible_and_position_indexed():
    rng = np.random.default_rng(1)
    a, b = _unit(rng, (2, 768)), _unit(rng, (6, 768))
    nm = NoiseModel(sigma_alpha=0.07, sigma_log_mu=0.1, sigma_vth=0.1, frozen_variation=True, variation_seed=5)
    dd = _dd(nm)
    m1 = dd.pair_matrix(a, b)
    m2 = dd.pair_matrix(a, b)
    np.testing.assert_array_equal(m1, m2)                    # same cells, same reads
    np.testing.assert_array_equal(dd.pair_matrix(a, b[:3]), m1[:, :3])   # row n keeps its cells
    fresh = _dd(NoiseModel(sigma_alpha=0.07, sigma_log_mu=0.1, sigma_vth=0.1,
                           frozen_variation=True, variation_seed=5))
    np.testing.assert_array_equal(fresh.pair_matrix(a, b), m1)          # independent of call history
    other = _dd(NoiseModel(sigma_alpha=0.07, sigma_log_mu=0.1, sigma_vth=0.1,
                           frozen_variation=True, variation_seed=6))
    assert not np.array_equal(other.pair_matrix(a, b), m1)              # seed selects the die


def test_zero_sigma_frozen_equals_clean():
    rng = np.random.default_rng(2)
    a, b = _unit(rng, (1, 768)), _unit(rng, (4, 768))
    clean = _dd(None).pair_matrix(a, b)
    frozen0 = _dd(NoiseModel(frozen_variation=True, variation_seed=1)).pair_matrix(a, b)
    np.testing.assert_array_equal(clean, frozen0)


def test_table_statistics():
    fz = FrozenCellVariation(seed=9)
    x = fz.field("gain", 0.2, (1, 400, 768))
    assert abs(float(np.std(x)) - 0.2) < 0.005 and abs(float(np.mean(x))) < 0.005
    y = fz.field("vth_l", 1.0, (400, 768)); z = fz.field("vth_r", 1.0, (400, 768))
    assert abs(float(np.corrcoef(y.ravel(), z.ravel())[0, 1])) < 0.01   # branches independent


def test_fefet_frozen_variation_is_fixed_and_changes_output():
    from aimdc_nvm_device_distance import NvmDeviceDistance, NoiseModel as NvmNoise
    rng = np.random.default_rng(3)
    q, c = _unit(rng, (2, 768)), _unit(rng, (16, 768))
    kw = dict(reduce="linear_sum", v_ml=1.4)
    sc = "h_aimdc_sim/fefet_ref_fit_smu.fs1p31.scaler.json"
    clean = NvmDeviceDistance("h_aimdc_sim/fefet_ref_fit_smu.json", sc, **kw).pair_matrix(q, c)
    nm = NvmNoise(sigma_alpha=0.05, sigma_log_mu=0.05, sigma_vth=0.02, frozen_variation=True, variation_seed=2)
    dd = NvmDeviceDistance("h_aimdc_sim/fefet_ref_fit_smu.json", sc, noise=nm, **kw)
    m1, m2 = dd.pair_matrix(q, c), dd.pair_matrix(q, c)
    np.testing.assert_array_equal(m1, m2)
    assert not np.allclose(m1, clean)
