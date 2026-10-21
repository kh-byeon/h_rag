#!/usr/bin/env bash
# =============================================================================
# n=3500 distractor run -- the decisive test of the quantization claim.
#
# QUESTION (pre-registered, decided before the run):
#   Does MMR degrade LESS than L2 from int8 to int4?  i.e. is the
#   difference-in-differences significant?
#
#   At n=1000:  EM DiD +0.0180 [-0.0025, +0.0385]  P = 0.086   ns
#               F1 DiD +0.0152 [-0.0037, +0.0337]  P = 0.115   ns
#   n for 80% power at those effects: EM 2,660 / F1 3,149.  n=3500 covers both
#   IF the observed effect is the true one -- see docs/RESULTS_q1000.md §5.
#
# DECISION RULE, fixed in advance:
#   * EM or F1 DiD 95% CI excludes 0  -> the "degrades less under quantization"
#     claim is supported; it becomes the paper's main result.
#   * neither excludes 0              -> STOP. Do not run a larger n, do not try
#     further reformulations (five have already failed, see §4/§4b). Fall back to
#     the operating-point framing in §4 and write the paper on what is already
#     established.
#
#   fp16 is deliberately NOT run: it CPU-offloads on this machine (~14x slower,
#   see docs/ANALYSIS_q1000.md §4) and int8->int4 is an equally good contrast
#   (DiD +0.0155 vs +0.0152, n@80% 3,122 vs 3,149).
#
#     bash scripts/run_q3500_did.sh
#
# Safe to re-run; every step is guarded.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

PY="${PY:-python}"
NQ="${NQ:-3500}"
BASE_RUN="${BASE_RUN:-sweep_exports/20260826_223418_mmr-capacitor_centroid-nvm}"
STATE="$ROOT/.q3500_run_id"
mkdir -p logs figures

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --------------------------------------------------- 0) preserve the n=1000 build --
# processed_hotpot/ and offline_data_hotpot/ are hardcoded paths (see
# online_accelerator_device.load_dataset_for_domain), so the 3500-question build
# must overwrite them. Keep a copy so the published n=1000 run stays reproducible.
if [ ! -d processed_hotpot_q1000 ]; then
  say "step 0: backing up the n=1000 build"
  cp -r processed_hotpot processed_hotpot_q1000
  cp -r offline_data_hotpot offline_data_hotpot_q1000
  echo "  processed_hotpot_q1000/ and offline_data_hotpot_q1000/ written."
  echo "  To restore: rm -rf processed_hotpot offline_data_hotpot && \\"
  echo "              cp -r processed_hotpot_q1000 processed_hotpot && \\"
  echo "              cp -r offline_data_hotpot_q1000 offline_data_hotpot"
else
  say "step 0 SKIPPED -- n=1000 build already backed up"
fi

# ------------------------------------------------------- 1) preprocess + index --
CUR=$("$PY" -c "import json;print(len(json.load(open('processed_hotpot/hotpot_queries.json'))))")
if [ "$CUR" != "$NQ" ]; then
  say "step 1: preprocessing to $NQ questions (records[:$NQ] -- keeps the first 1000 in order)"
  "$PY" preprocess_hotpot.py --input hotpot_dev_distractor_v1.json \
    --out_dir processed_hotpot --max_questions "$NQ" 2>&1 | tee logs/q3500_preprocess.log
  say "step 2: rebuilding the offline index over the larger corpus"
  "$PY" off_line_prepper.py --domain hotpot 2>&1 | tee logs/q3500_index.log
else
  say "steps 1-2 SKIPPED -- processed_hotpot already holds $NQ queries"
fi
GOT=$("$PY" -c "import json;print(len(json.load(open('processed_hotpot/hotpot_queries.json'))))")
[ "$GOT" = "$NQ" ] || { echo "expected $NQ queries, got $GOT" >&2; exit 1; }

# ------------------------------------------------------------- 3) retrieval --
if [ -f "$STATE" ] && [ -f "sweep_exports/$(cat "$STATE")/retrieval_$(cat "$STATE").jsonl" ]; then
  RUN_ID="$(cat "$STATE")"; say "step 3 SKIPPED -- retrieval done: $RUN_ID"
else
  say "step 3: retrieval sweep (distractor, n=$NQ, one combo)"
  BEFORE="$(mktemp)"; ls sweep_exports | sort > "$BEFORE"
  "$PY" sweep_eval.py --domain hotpot --num_queries "$NQ" \
    --retrieval_scope distractor \
    --lambdas 0.7 --gammas 0.7 --tau_dups=-1.0 --top_ks 5 \
    --centroid_mode nvm \
    --centroid_fit_path h_aimdc_sim/fefet_ref_fit_smu.json \
    --centroid_reduce linear_sum --centroid_v_ml 1.4 \
    --mmr_mode capacitor --device_v_min -3.0 --device_v_max 3.0 \
    2>&1 | tee logs/q3500_retrieval.log
  RUN_ID="$(ls sweep_exports | sort | comm -13 "$BEFORE" - | tail -1)"; rm -f "$BEFORE"
  [ -n "$RUN_ID" ] || { echo "could not identify the new run_id" >&2; exit 1; }
  echo "$RUN_ID" > "$STATE"
  cat > "sweep_exports/$RUN_ID/run_flags.txt" <<FLAGS
run_id=$RUN_ID
generated_by=scripts/run_q3500_did.sh
date=$(date -Iseconds)
purpose=powering the int8->int4 difference-in-differences (see the script header)
retrieval_scope=distractor   num_queries=$NQ
grid=lambda:0.7 tau_dup:-1.0 gamma:0.7 top_k:5   anchor_protect_n=0
centroid_mode=nvm  centroid_fit_path=h_aimdc_sim/fefet_ref_fit_smu.json  centroid_reduce=linear_sum  centroid_v_ml=1.4
mmr_mode=capacitor  device_v_min=-3.0  device_v_max=3.0  standardize_mmr_terms=True
quantizations=int8,int4  (fp16 skipped: CPU offload, see docs/ANALYSIS_q1000.md)
FLAGS
fi
RJ="sweep_exports/$RUN_ID/retrieval_$RUN_ID.jsonl"

# ------------------------------------------------------- 4) nesting check --
say "step 4: is the n=1000 result a subset of this run?"
"$PY" scripts/check_run_nesting.py "$BASE_RUN" "sweep_exports/$RUN_ID" \
  2>&1 | tee logs/q3500_nesting.log || {
  echo
  echo "  NOTE: not nested. The run is still valid on its own -- the DiD below is"
  echo "  computed within this run only and does not depend on nesting. But do not"
  echo "  describe the n=1000 result as a subset of it in the paper."
}

# ----------------------------------------------------------- 5) generation --
for Q in int8 int4; do
  OUT="llm_out_q3500_$Q"
  if [ -n "$(ls "$OUT"/combos/llm_combo_*.json 2>/dev/null || true)" ]; then
    say "step 5 [$Q] SKIPPED -- $OUT/combos already populated"; continue
  fi
  say "step 5 [$Q]: generation ($((NQ*2)) prompts; the first 1000 queries are already cached)"
  "$PY" scripts/run_generation_sweep.py --retrieval_jsonl "$RJ" \
    --llm_output_root "$OUT" --quantization "$Q" \
    --cache "cache/gen_llama3-8b_$Q.sqlite" --chunk 1 \
    2>&1 | tee "logs/q3500_gen_$Q.log"
done

# -------------------------------------------------------------- 6) scoring --
COMBOS=(llm_out_q3500_int8/combos:int8 llm_out_q3500_int4/combos:int4)
say "step 6a: EM / F1"
"$PY" scripts/collect_em_f1.py --combos_dir "${COMBOS[@]}" \
  --out figures/em_f1_q3500.csv 2>&1 | tee logs/q3500_emf1.log
say "step 6b: faithfulness"
"$PY" scripts/score_faithfulness_relevance.py --combos_dir "${COMBOS[@]}" \
  --top_k 5 --out figures/faith_rel_q3500.csv 2>&1 | tee logs/q3500_faith.log

# ------------------------------------------------------------- 7) the test --
say "step 7: the pre-registered difference-in-differences"
"$PY" scripts/analyze_did_q3500.py --run_id "$RUN_ID" 2>&1 | tee logs/q3500_did.log
