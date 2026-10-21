#!/usr/bin/env bash
# =============================================================================
# n=1000 distractor run  --  the "make faithfulness significant" run of
# docs/HANDOFF_mmr_evaluation.md section 7.
#
#   scope      : distractor  (HotpotQA standard; isolates the reranker)
#   grid       : ONE combo, lambda=0.7 tau=-1.0 gamma=0.7 top_k=5
#   queries    : 1000 (all of processed_hotpot/hotpot_queries.json, in order)
#   device path: FeFET centroid (linear_sum, V_ML=1.4) + IGZO capacitor MMR
#
# Run it from the repo root, inside WSL, with the project venv active:
#
#     bash scripts/run_q1000_distractor.sh
#
# It is safe to re-run after any interruption -- see "Resumability" below.
# =============================================================================
#
# Resumability
# ------------
#   step 0 (prime)      : idempotent; guarded by cache/.primed_<quant>
#   step 1 (retrieval)  : skipped once .q1000_run_id exists and its JSONL is
#                         present. Delete that file to force a fresh retrieval.
#   step 2 (generation) : run_generation_sweep.py resumes at *combo* granularity
#                         -- and this run has only ONE combo, so a crash at
#                         query 900/1000 leaves no combo JSON behind. The real
#                         safety net is the prompt cache: gen_cache.put()
#                         commits every generation immediately under WAL, so a
#                         restart pays only the 8B model load and then replays
#                         from sqlite. DO NOT disable the cache.
#   step 3 (scoring)    : cheap, always re-runs.
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

PY="${PY:-python}"
STATE="$ROOT/.q1000_run_id"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR" figures cache

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- preflight --
say "preflight"
for f in processed_hotpot/hotpot_queries.json \
         h_aimdc_sim/fefet_ref_fit_smu.json \
         h_aimdc_sim/igzo_fit.json \
         h_aimdc_sim/igzo_fit.k.json; do
  [ -f "$ROOT/$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
[ -d "$ROOT/offline_data_hotpot" ] || { echo "MISSING: offline_data_hotpot/" >&2; exit 1; }
NQ=$("$PY" -c "import json;print(len(json.load(open('processed_hotpot/hotpot_queries.json'))))")
[ "$NQ" = "1000" ] || { echo "expected 1000 queries, found $NQ" >&2; exit 1; }
echo "ok: 1000 queries, offline index, FeFET + IGZO fits all present"

# ------------------------------------------------- 0) prime the prompt cache --
# The existing cache DBs hold only the *cross* sweep (2,996 rows each); the
# 135-combo distractor run was generated WITHOUT --cache, so its answers are on
# disk but invisible to the cache. Re-inserting them is lossless (greedy
# decoding => prompt determines answer exactly), and since the n=100 query set
# is random.sample(all_1000, 100, seed=42) -- a SUBSET of these 1000 -- roughly
# 10% of this run's generations come back as exact hits for free.
# Verified: prompts rebuilt from a retrieval JSONL match the stored prompts
# 200/200 at this combo.
for Q in fp16 int8 int4; do
  MARK="cache/.primed_$Q"
  if [ -f "$MARK" ]; then
    say "step 0 [$Q] SKIPPED -- cache already primed"
    continue
  fi
  if [ ! -d "llm_out_$Q/combos" ]; then
    echo "  no llm_out_$Q/combos -- nothing to prime"; continue
  fi
  say "step 0 [$Q]: prime cache from the existing 135-combo distractor run"
  "$PY" scripts/prime_gen_cache.py \
    --combos_dir "llm_out_$Q/combos" \
    --cache "cache/gen_llama3-8b_$Q.sqlite" \
    2>&1 | tee "$LOGDIR/q1000_prime_$Q.log"
  date -Iseconds > "$MARK"
done

# ------------------------------------------------------- 1) retrieval sweep --
if [ -f "$STATE" ] && [ -f "sweep_exports/$(cat "$STATE")/retrieval_$(cat "$STATE").jsonl" ]; then
  RUN_ID="$(cat "$STATE")"
  say "step 1 SKIPPED -- retrieval already done: $RUN_ID"
else
  say "step 1: retrieval (distractor, n=1000, 1 combo)"
  BEFORE="$(mktemp)"; ls sweep_exports 2>/dev/null | sort > "$BEFORE" || true

  "$PY" sweep_eval.py --domain hotpot --num_queries 1000 \
    --retrieval_scope distractor \
    --lambdas 0.7 --gammas 0.7 --tau_dups=-1.0 --top_ks 5 \
    --centroid_mode nvm \
    --centroid_fit_path h_aimdc_sim/fefet_ref_fit_smu.json \
    --centroid_reduce linear_sum --centroid_v_ml 1.4 \
    --mmr_mode capacitor --device_v_min -3.0 --device_v_max 3.0 \
    2>&1 | tee "$LOGDIR/q1000_retrieval.log"

  RUN_ID="$(ls sweep_exports | sort | comm -13 "$BEFORE" - | tail -1)"
  rm -f "$BEFORE"
  [ -n "$RUN_ID" ] || { echo "could not identify the new run_id" >&2; exit 1; }
  echo "$RUN_ID" > "$STATE"
  echo "run_id = $RUN_ID"

  # sweep_eval's run_manifest.json records NEITHER --centroid_fit_path NOR
  # --centroid_reduce (handoff section 9), so the FeFET attribution is not
  # recoverable from the artifacts alone. Pin it here, next to the data.
  cat > "sweep_exports/$RUN_ID/run_flags.txt" <<FLAGS
run_id=$RUN_ID
generated_by=scripts/run_q1000_distractor.sh
date=$(date -Iseconds)
retrieval_scope=distractor
num_queries=1000
grid=lambda:0.7 tau_dup:-1.0 gamma:0.7 top_k:5
nprobe=32 (default; irrelevant under distractor scope -- FeFET runs as dead compute)
centroid_mode=nvm
centroid_fit_path=h_aimdc_sim/fefet_ref_fit_smu.json   <-- FeFET path, NOT the igzo_fit_v3 default
centroid_reduce=linear_sum                             <-- FeFET path, NOT the 'sum' default
centroid_v_ml=1.4
mmr_mode=capacitor
device_fit_path=h_aimdc_sim/igzo_fit.json              (default; IGZO capacitor MMR)
device_scale_k_path=h_aimdc_sim/igzo_fit.k.json        (default)
device_v_min=-3.0 device_v_max=3.0
standardize_mmr_terms=True (default on)
FLAGS
fi

RJ="sweep_exports/$RUN_ID/retrieval_$RUN_ID.jsonl"
[ -f "$RJ" ] || { echo "MISSING retrieval jsonl: $RJ" >&2; exit 1; }

# ----------------------------------------------------------- 2) generation --
for Q in fp16 int8 int4; do
  OUT="llm_out_q1000_$Q"
  if [ -n "$(ls "$OUT"/combos/llm_combo_*.json 2>/dev/null || true)" ]; then
    say "step 2 [$Q] SKIPPED -- $OUT/combos already populated"
    continue
  fi
  say "step 2 [$Q]: generation  (2000 prompts, minus cache hits)"
  "$PY" scripts/run_generation_sweep.py \
    --retrieval_jsonl "$RJ" \
    --llm_output_root "$OUT" \
    --quantization "$Q" \
    --cache "cache/gen_llama3-8b_$Q.sqlite" \
    --chunk 1 \
    2>&1 | tee "$LOGDIR/q1000_gen_$Q.log"
done

# -------------------------------------------------------------- 3) scoring --
COMBOS=(llm_out_q1000_fp16/combos:fp16
        llm_out_q1000_int8/combos:int8
        llm_out_q1000_int4/combos:int4)

say "step 3a: faithfulness / answer-relevance"
"$PY" scripts/score_faithfulness_relevance.py \
  --combos_dir "${COMBOS[@]}" \
  --top_k 5 --out figures/faith_rel_q1000.csv \
  2>&1 | tee "$LOGDIR/q1000_faith.log"

say "step 3b: EM / F1"
"$PY" scripts/collect_em_f1.py \
  --combos_dir "${COMBOS[@]}" \
  --out figures/em_f1_q1000.csv \
  2>&1 | tee "$LOGDIR/q1000_emf1.log"

say "done"
cat <<DONE

  run_id          : $RUN_ID
  retrieval       : $RJ
  generations     : llm_out_q1000_{fp16,int8,int4}/combos/
  faithfulness    : figures/faith_rel_q1000.csv
  EM / F1         : figures/em_f1_q1000.csv
  provenance      : sweep_exports/$RUN_ID/run_flags.txt

  Next: cluster by query_idx BEFORE testing (handoff section 6). n=1000, not
  n = rows. Note query_idx is a positional index into the sampled eval_pairs,
  so it is NOT comparable to query_idx in the n=100 runs -- join on the query
  text if you ever need to line the two up.

DONE
