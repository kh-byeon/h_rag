# h_rag — U-AIMDC simulation pipeline

Code for **"Monolithic integration of ferroelectric and dielectric capacitors on a unified oxide-TFT platform for in-memory distance computing"**
(K. A. Byun, H. S. Choi, C. Joe, D. H. Han, M. H. Park and S. Kim, Seoul National University).

This repository contains the system-level simulation of unified analog in-memory distance
computing (U-AIMDC) for retrieval-augmented generation (RAG): IVF/PQ index construction, the
measurement-calibrated cell models of the static tier (FeCAP-coupled FeFET pair, centroid search)
and of the dynamic tier (DeCAP-coupled IGZO 4T2C cell, on-the-fly maximal-marginal-relevance
re-ranking), quantized-LLM answer generation, scoring, the statistical analysis and the simulation
ablation reported in the paper. Device measurement data, the per-question results and the
figure-plotting scripts live in the companion data repository
**[h_aimdc_data](https://github.com/kh-byeon/h_aimdc_data)** (tag `v1.1.0`).

Version used in the paper: **v1.1.0** (git tag `v1.1.0`). v1.0.0 is the version cited in the
original submission; the changes are listed in `CHANGELOG.md`.

### Naming

The repository, package and file names keep the project's original name, H-AIMDC
(heterogeneous AIMDC), so that paths and run identifiers stay stable. In the paper the platform
is called U-AIMDC, and the two storage roles are named after their capacitors:

| Paper | Code |
|---|---|
| U-AIMDC | H-AIMDC, `h_aimdc_sim`, `aimdc_*` |
| FeCAP cell (static tier, centroids) | FeFET, `nvm`, `aimdc_nvm_device_distance`, `--centroid_mode nvm` |
| DeCAP cell (dynamic tier, selected set) | IGZO capacitor, `capacitor`, `aimdc_device_distance`, `--mmr_mode capacitor` |

## Repository layout

| Path | Role |
|---|---|
| `download_hotpot.py`, `preprocess_hotpot.py` | Fetch HotpotQA (dev, distractor) and build the local corpus/queries JSON |
| `preprocess_cuad.py` | Build the CUAD corpus/queries JSON from `CUAD_v1/` |
| `off_line_prepper.py` | Offline indexing: embedding (multi-qa-mpnet-base-dot-v1, d = 768, L2-normalised), 128 IVF centroids, product quantisation into 8 sub-vectors |
| `online_accelerator_device.py` | Online retrieval: centroid search on the static tier, candidate streaming, on-the-fly MMR with the selected set resident in the dynamic tier, write/readout statistics and memory-traffic accounting |
| `aimdc_nvm_device_distance/` | Analytical FeFET (2-FeFET complementary cell) match-line distance model for the static tier |
| `aimdc_device_distance/` | Analytical IGZO-capacitor (4T2C) match-line distance model for the dynamic tier, with optional noise mechanisms (`noise_configs/`) |
| `aimdc_variation.py` | Frozen device-to-device variation (per-cell coupling ratio, gain and threshold offset drawn once and reused on every read), used by both device models when `frozen_variation` is set in a noise config |
| `h_aimdc_sim/` | Cell-level models and calibration against the measured devices: `fefet_ref_model.py`, `fefet_vth_calibration.py`, `igzo_powerlaw_model.py`, the fitted parameter files (`fefet_ref_fit_smu.json`, `igzo_fit.json`, `igzo_fit.k.json`, `nvm_operating_point.json`), and the Xyce/Verilog-A decks used to validate the analytical cell model (`igzo_tft.va`, `sim_*.sp`, `Dockerfile`) |
| `sweep_eval.py` | Retrieval sweep driver (λ, τ_dup, γ, top-k grid) producing `sweep_exports/<run_id>/retrieval_<run_id>.jsonl` |
| `llm_generator.py`, `scripts/run_generation_sweep.py` | Answer generation with Meta-Llama-3-8B-Instruct at fp16 / int8 / int4 (bitsandbytes), greedy decoding, resumable |
| `postprocess.py`, `run_eval_from_generation.py`, `evaluation/` | Local, air-gapped post-processing of chunk references and the evaluation framework (answer EM/F1, supporting-passage recall, coverage, faithfulness/relevance via NLI) |
| `scripts/` | Reproduction scripts (see below): `run_q1000_distractor.sh`, `analyze_q1000.py`, `run_write_stats*.sh`, `analyze_write_stats.py`, `make_fig2_counts.py`, `make_fig2_precision.py`, `make_paper_tables.py`, `analyze_did_q3500.py`, `calibrate_scale_k.py`, `validate_analytical_vs_xyce.py`; simulation ablation: `run_ablation_q1000.sh`, `analyze_ablation.py`, `make_ablation_noise_configs.py`, `dump_search_text_embeddings.py` |
| `sweep_exports/tol_run/` | Stored-value precision sweep proxy (`tol_probe.py`: IVF probe + top-50 MMR under injected Gaussian write noise) used by `make_fig2_precision.py` |
| `tests/` | Unit tests for the device-distance models, MMR scoring units, write statistics and noise configs (`pytest tests`) |
| `cuad/` | The variant of the pipeline files used for the CUAD intra-document experiment (Fig. 5a, Supplementary Note 5); see `cuad/README.md` |

## Requirements

* Linux (developed and run under Ubuntu on WSL2), Python ≥ 3.10.
* An NVIDIA GPU is needed only for answer generation and NLI scoring
  (Meta-Llama-3-8B-Instruct: ~16 GB of GPU memory at fp16, ~6 GB at int4).
  Retrieval, the device models, the write-statistics runs and the analysis scripts run on CPU.
* Access to the gated `meta-llama/Meta-Llama-3-8B-Instruct` weights on Hugging Face
  (accept the Meta Llama 3 licence, then `huggingface-cli login`).
* Optional: Docker, for the Xyce/OpenVAF validation of the analytical cell model
  (`run_xyce.sh`, `h_aimdc_sim/docker-compose.xyce.yml`).

```bash
git clone https://github.com/kh-byeon/h_rag.git
cd h_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # ~5 min; torch wheel dominates
```

## Data

| Dataset / model | Source |
|---|---|
| HotpotQA dev set, distractor setting (7,405 questions) | https://hotpotqa.github.io/ (`download_hotpot.py` fetches the Hugging Face mirror `hotpotqa/hotpot_qa`) |
| CUAD v1 (510 contracts, 41 clause categories) | https://www.atticusprojectai.org/cuad — place `CUAD_v1.json` and `full_contract_txt/` under `CUAD_v1/` (or set `CUAD_V1_ROOT`) |
| Embedding encoder | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| Generator | `meta-llama/Meta-Llama-3-8B-Instruct` |
| NLI scorer (faithfulness, aspect coverage) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |

```bash
# HotpotQA: first 1,000 questions of the dev/distractor split, in their original order
python download_hotpot.py --config distractor --split validation --out hotpot_dev_distractor_v1.json
python preprocess_hotpot.py --input hotpot_dev_distractor_v1.json --out_dir processed_hotpot --max_questions 1000
python off_line_prepper.py --domain hotpot --output_dir offline_data_hotpot
# the write-statistics scripts look for the same n = 1,000 build under these names:
ln -s processed_hotpot processed_hotpot_q1000 && ln -s offline_data_hotpot offline_data_hotpot_q1000

# CUAD
python preprocess_cuad.py
python off_line_prepper.py --domain cuad --output_dir offline_data_cuad
```

## Reproducing the paper

All commands run from the repository root with the virtual environment active.
The device fits shipped in `h_aimdc_sim/` are the calibrations used in the paper:

* `fefet_ref_fit_smu.json` — FeFET reference-curve (LUT) model built by `h_aimdc_sim/fefet_ref_model.py`
  from the measured FeFET transfer curve (the source file and its SHA-256 are recorded inside the JSON);
  `fefet_vth_calibration.json` holds the rigid-V_th calibration of the programmed states
  (`fefet_vth_calibration.py`). The static-tier operating point used in the paper is passed on the
  command line (`--centroid_reduce linear_sum --centroid_v_ml 1.4`, see `scripts/run_q1000_distractor.sh`).
* `igzo_fit.json` — IGZO-TFT power-law model fitted to `h_aimdc_sim/igzo_measure.csv`
  (`python h_aimdc_sim/aimdc_sim_core.py fit --csv h_aimdc_sim/igzo_measure.csv --out h_aimdc_sim/igzo_fit.json`);
  `igzo_fit.k.json` — the scale factor that maps the array distance onto the software-L2 scale
  (`python scripts/calibrate_scale_k.py --offline_dir offline_data_hotpot --device_fit_path h_aimdc_sim/igzo_fit.json --out_json h_aimdc_sim/igzo_fit.k.json`).

```bash
# 1. HotpotQA, n = 1,000: retrieval (both arms) -> generation at fp16/int8/int4 -> scoring
#    FeFET centroid search (linear_sum, V_ML = 1.4 V) + IGZO-capacitor on-the-fly MMR
#    lambda = 0.7, tau_dup = -1.0 (z-standardised), gamma = 0.7, top_k = 5
bash scripts/run_q1000_distractor.sh                 # ~hours on one GPU; resumable

# 2. Every number in Fig. 5b-c and the Methods statistics -> figures/q1000_metrics.xlsx
#    (paired bootstrap over questions, 20,000 resamples, fixed seed; McNemar; Bonferroni; DiD)
python scripts/analyze_q1000.py --rebuild

# 3. Write / readout statistics of the dynamic tier (Fig. 2, Fig. 5d, Methods)
bash scripts/run_write_stats_ksweep.sh               # k = 5, 10, 20; distractor and cross scopes
python scripts/make_fig2_counts.py --out figures/fig2

# 4. Stored-value precision requirement per tier (Fig. 2c)
python scripts/make_fig2_precision.py --out figures/fig2

# 5. Tables and the pre-registered difference-in-differences test
python scripts/make_paper_tables.py
python scripts/analyze_did_q3500.py

# 6. Simulation ablation on the same 1,000 questions (Supplementary Note 6.4, Supplementary Table 14):
#    digital L2, digital MMR with the identical procedure (--mmr_mode software_std), the calibrated
#    average device, and frozen DeCAP device-to-device variation at 0.5x/1x/2x and per mechanism,
#    five simulated dies each. Retrieval only, ~1 min per run on CPU; runs inside ws_q1000/.
bash scripts/run_ablation_q1000.sh
python scripts/analyze_ablation.py ablation/manifest.tsv --out ablation/ablation_q1000 \
       --reference sweep_exports/<run_id of step 1> --export_perq ablation/ablation_q1000_perquestion.csv.gz
#    Without the sentence encoder, replay the query embeddings written by
#    scripts/dump_search_text_embeddings.py:  QEMB=sweep_exports/query_emb_q1000_searchtext.npz bash ...

# 7. Unit tests (~1 min, CPU)
pytest tests
```

The plots themselves are produced by the scripts in the data repository
(`plot_fig2_tiers.py`, `plot_fig2_precision.py`, `plot_q1000_figures.py`, `plot_fig5d_operands.py`,
`plot_fig5e_memory.py`), which read the CSV/JSON outputs of steps 2-4.

### Demo / smoke test

```bash
NQ=20 bash scripts/run_write_stats_ksweep.sh         # retrieval only, 20 questions, a few minutes on CPU
python scripts/make_fig2_precision.py --smoke        # 2 sigmas, 1 seed
pytest tests
```

### CUAD intra-document experiment (Fig. 5a, Supplementary Note 5)

The CUAD run used the pipeline variant in `cuad/` (clause-level chunking, intra-document
retrieval, the four-step legal prompt, local chunk-reference resolution and aspect
coverage scoring). Copy the files in `cuad/` over the top-level ones of the same name, then

```bash
python sweep_eval.py --domain cuad --num_queries 50 --lambdas 0.5 --gammas 0.2 --tau_dups=-1.0 --top_ks 10 \
       --mmr_mode capacitor --centroid_mode nvm
python scripts/run_generation_sweep.py --retrieval_jsonl sweep_exports/<run_id>/retrieval_<run_id>.jsonl \
       --llm_output_root llm_out_<quant> --quantization <fp16|int8|int4>
python run_eval_from_generation.py --normalized_json generation_normalized_<quant>.json --output_dir evaluation_outputs
```

These are the parameters of the reported CUAD run (k = 10, λ = 0.5, γ = 0.2, τ_dup = −1.0; run id
`20260811_181301_mmr-capacitor_centroid-nvm`; v1.0.0 of this README listed the HotpotQA values by mistake).
The centroid fit flags of that run were not recorded in its manifest.

## Citation

If you use this code, please cite the paper (preprint/DOI to be added) and this
repository (`CITATION.cff`).

## License

MIT License (see `LICENSE`). The datasets and models listed above are subject to their own
licences (CUAD: CC BY 4.0; HotpotQA: CC BY-SA 4.0; Meta Llama 3: Meta Llama 3 Community License).
