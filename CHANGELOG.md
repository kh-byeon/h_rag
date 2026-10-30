# Changelog

## v1.1.0 — 2026-10-30 (revised manuscript)

Paper: "Monolithic integration of ferroelectric and dielectric capacitors on a unified oxide-TFT platform for in-memory distance computing" (revised title; the original submission was titled "Diversity-aware
retrieval-augmented generation acceleration at the edge enabled by FeFET-IGZO heterogeneous analog
in-memory distance computing").

Added
- Simulation ablation (Supplementary Note 6.4, Supplementary Table 14): frozen device-to-device variation
  (`aimdc_variation.py`; `frozen_variation` and `variation_seed` in both noise models), a digital arm
  with the identical re-ranking procedure (`--mmr_mode software_std`), `scripts/run_ablation_q1000.sh`,
  `scripts/analyze_ablation.py`, `scripts/make_ablation_noise_configs.py`, `tests/test_frozen_variation.py`.
- `sweep_eval.py --query_emb_map` and `scripts/dump_search_text_embeddings.py`: replay query embeddings
  from disk exactly as the pipeline encodes them (126 of the 1,000 HotpotQA questions are encoded as
  "keyword. question").

Changed
- README, CITATION: title and terminology of the revised manuscript (U-AIMDC; FeCAP and DeCAP cells).
  Repository, package and file names are unchanged (see "Naming" in the README).
- README: the CUAD reproduction command now lists the parameters of the reported run (k = 10, λ = 0.5,
  γ = 0.2, τ_dup = −1.0); v1.0.0 listed the HotpotQA values. Supplementary Note numbers follow the
  revised Supplementary Information (CUAD case study: Note 5).

Unchanged
- Retrieval, device-model and generation code paths used for the v1.0.0 results. With variation off, the
  reported HotpotQA run (`20260826_223418`) is reproduced for 1,000 of 1,000 questions (L2 and MMR lists).

## v1.0.0 — 2026-10-21

Version cited in the original submission.
