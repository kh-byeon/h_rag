# h_aimdc_sim

## Scope

AIMDC is the MMR diversity accelerator only. In the online pipeline:

- `d(q, C)`: PQ-LUT (software L2 in this codebase). Used for anchor selection and
  as the relevance term of the MMR score.
- `min_d(C, S)`: AIMDC. Candidate C on V_SL, anchor+selected S on V_PRG,
  parallel match-line current readout.

When **scale_k** is not loaded, the MMR score mixes units (software L2 vs
AIMDC raw). After calibration (recommended), diversity is scaled to the
software-L2 scale while raw AIMDC values remain in dumps.

Scaler is fit on the PQ-reconstructed embedding distribution (single scaler,
same distribution on both V_SL and V_PRG sides).

## Calibration (scale_k)

AIMDC distance (log-sum-current) is offset/scale-different from software L2.
The calibration constant k maps AIMDC values to the software-L2 scale so that:

- `tau_dup` can be set in software-L2 units (the value that worked in the software
  baseline remains valid for AIMDC)
- λ recovers its textbook meaning (0.5 = balanced, 0.0 = pure diversity,
  1.0 = pure relevance)

Measurement (offline, one time):

    python scripts/calibrate_scale_k.py \
        --offline_dir ./offline_data_cuad \
        --device_fit_path h_aimdc_sim/igzo_fit.json \
        --out_json h_aimdc_sim/igzo_fit.k.json

Apply at inference:

    python online_accelerator_device.py ... \
        --use_device_distance \
        --device_fit_path h_aimdc_sim/igzo_fit.json \
        --device_scale_k_path h_aimdc_sim/igzo_fit.k.json

**Scope**: k affects only the MMR score formula and τ comparison. Raw AIMDC
distances are preserved verbatim in dumps (`min_dist_c_s_raw`,
`min_dist_c_s_used`). Scaled values are added as `min_dist_c_s_scaled`.
