#!/usr/bin/env python3
"""Write the DeCAP device-to-device variation configs for the four-arm ablation.

    python scripts/make_ablation_noise_configs.py [--seeds 1,2,3,4,5] [--out noise_configs/ablation]

Every config sets ``frozen_variation`` so that each cell keeps its gain, coupling ratio and
threshold offset for every read (aimdc_variation.py). Values at 1x come from the IGZO 4T2C
array measurements (h_aimdc_data/4t2c_array/4t2c_distance_paper_20260904.xlsx); the
conversions are written out below so they can be checked and changed.

Coupling ratio (sigma_alpha, relative)
    per-cell alpha from v0(-w) - v0(+w) = 2w/alpha, w = 0.35 V (sheet Fig3c_alpha_percell):
    six working cells, mean 0.566, s.d. 0.038  ->  CV = 0.067. The seventh cell (alpha = 1.52)
    belongs to the defective row and is excluded.

Gain (sigma_log_mu, log10 of the current prefactor)
    per-cell curvature kappa (sheet Fig7_cell_cal, defective row excluded, n = 18):
    s.d. of ln(kappa) = 0.31. kappa scales as gain x alpha^2, so the alpha share
    (2 x 0.067 = 0.13 in ln) is removed: sqrt(0.31^2 - 0.13^2) = 0.28 in ln = 0.12 in log10.

Threshold offset (sigma_vth, model volts, per branch)
    notch position minus the nominal map v0 = -w/alpha_med + b (Fig7_cell_cal, n = 18):
    s.d. 75 mV in the physical query coordinate, whose span in that measurement was 2.06 V
    (x in [-1.023, +1.038] V). The simulator maps embeddings onto a 6 V span
    (--device_v_min -3 --device_v_max 3), so the notch offset in model volts is
    0.075 x 6 / 2.06 = 0.218 V. A notch moves by half the difference of the two branch
    thresholds, so the per-branch threshold s.d. is sqrt(2) x 0.218 = 0.31 V.

The three measured quantities come from 4-18 cells, so the scenarios scale them by 0.5, 1 and 2,
and three single-mechanism scenarios at 1x separate their effects. FeFET (centroid) variation
is not written: under the distractor scope the centroid search does not affect the candidate
pool, so it cannot change these results.
"""
from __future__ import annotations

import argparse
import json
import math
import os

ALPHA_CV = 0.067
GAIN_LN_TOTAL, ALPHA_LN_SHARE = 0.31, 2 * 0.067
GAIN_LOG10 = math.sqrt(GAIN_LN_TOTAL ** 2 - ALPHA_LN_SHARE ** 2) / math.log(10)
NOTCH_SD_PHYS, PHYS_SPAN, MODEL_SPAN = 0.075, 2.061, 6.0
VTH_BRANCH = math.sqrt(2) * NOTCH_SD_PHYS * MODEL_SPAN / PHYS_SPAN

BASE = {"sigma_alpha": ALPHA_CV, "sigma_log_mu": GAIN_LOG10, "sigma_vth": VTH_BRANCH}
SCENARIOS = {
    "all_0.5x": {k: 0.5 * v for k, v in BASE.items()},
    "all_1x": dict(BASE),
    "all_2x": {k: 2.0 * v for k, v in BASE.items()},
    "alpha_1x": {"sigma_alpha": BASE["sigma_alpha"]},
    "gain_1x": {"sigma_log_mu": BASE["sigma_log_mu"]},
    "offset_1x": {"sigma_vth": BASE["sigma_vth"]},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--out", default="noise_configs/ablation")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    os.makedirs(a.out, exist_ok=True)
    print(f"1x: sigma_alpha={BASE['sigma_alpha']:.4f}  sigma_log_mu={BASE['sigma_log_mu']:.4f}  "
          f"sigma_vth={BASE['sigma_vth']:.4f} V")
    for name, sig in SCENARIOS.items():
        for s in seeds:
            cfg = {"frozen_variation": True, "variation_seed": s, **{k: round(v, 5) for k, v in sig.items()}}
            path = os.path.join(a.out, f"decap_{name}_s{s}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=1)
    print(f"wrote {len(SCENARIOS) * len(seeds)} configs to {a.out}")


if __name__ == "__main__":
    main()
