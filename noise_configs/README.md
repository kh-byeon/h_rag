# Noise configs

Each JSON here maps onto `NoiseModel` fields. Fields not listed remain at
`NoiseModel` defaults (all off). To apply at runtime:

    python online_accelerator_device.py ... \
        --use_device_distance \
        --device_fit_path h_aimdc_sim/igzo_fit.json \
        --device_scale_k_path h_aimdc_sim/igzo_fit.k.json \
        --noise_config noise_configs/d2d.json

## Files

| file | enabled mechanisms |
|---|---|
| `clean.json` | (none; deterministic) |
| `d2d.json` | σ_vth, σ_logμ |
| `d2d_retention.json` | σ_vth, σ_logμ, retention |
| `full.json` | σ_vth, σ_logμ, retention, ADC 8-bit |

Mechanisms intentionally left off everywhere (explicit design choice):

- `ktc_C` — kT/C capacitor noise (would enable with a realistic C value)
- `sigma_alpha` — coupling-cap mismatch (covered by process, not primary focus)
- `flicker_sigma` — 1/f noise (Gaussian placeholder, no time correlation)
- `rtn_sigma` — RTN (placeholder, proper HMM TBD)
- `shot_enable` — shot noise (dominated by other sources here)

## Rationale for values

- `sigma_vth = 0.03` (30 mV): middle of IGZO TFT reported range (~20-50 mV).
- `sigma_log_mu = 0.05`: ~12% mobility std, conservative IGZO process.
- `retention_tau = 10 s, t_hold = 10 ms`: conservative; chosen so retention
  actually exercises the code path (long τ would be a no-op). Real IGZO
  retention is much longer; sweep t_hold to study the effect.
- `adc_bits = 8, fullscale = 1 µA`: per-bit-cell quantization at 3.9 nA LSB.
