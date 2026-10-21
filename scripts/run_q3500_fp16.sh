#!/usr/bin/env bash
set -e
cd /mnt/c/Users/khb/codes/H_RAG_cuad_v3.0_MMR_deviceModeling
source /home/khb/anaconda3/etc/profile.d/conda.sh
conda activate torch
RUN_ID=$(tr -d '[:space:]' < .q3500_run_id)
exec python scripts/run_generation_sweep.py \
  --retrieval_jsonl "sweep_exports/${RUN_ID}/retrieval_${RUN_ID}.jsonl" \
  --llm_output_root llm_out_q3500_fp16 \
  --quantization fp16 \
  --cache cache/gen_llama3-8b_fp16.sqlite \
  --chunk 1
