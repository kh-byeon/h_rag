#!/usr/bin/env bash
# =============================================================================
# IGZO-tier write / readout statistics at the paper's operating point.
#
# Two retrieval-only runs (no LLM), both on the n=1000 index build the paper
# reports (offline_data_hotpot_q1000 / processed_hotpot_q1000), one combo
# lambda=0.7 tau=-1.0 gamma=0.7 top_k=5, FeFET centroid + IGZO capacitor MMR:
#
#   R1  distractor scope            -- the accuracy setting  (pool ~10)
#   R2  cross scope, nprobe 32      -- the hardware setting  (pool ~2,400)
#
# then scripts/analyze_write_stats.py on each, and an identity check of R1
# against the published run 20260826_223418 (same flags, same build) proving
# the instrumentation did not change retrieval.
#
# Run from the repo root, inside WSL, with the project venv active:
#     bash scripts/run_write_stats.sh            # both runs
#     SCOPES="cross" bash scripts/run_write_stats.sh
#     NQ=50 bash scripts/run_write_stats.sh      # quick smoke
#
# The working tree currently holds the n=3500 build in processed_hotpot/ and
# offline_data_hotpot/ (those paths are hard-coded in the pipeline), so the
# runs execute inside ws_q1000/, a scratch dir of symlinks to the *_q1000
# copies. Nothing in the main tree is touched. Results land in
#     sweep_exports/write_stats/<scope>/<run_id>/   (+ write_stats_summary.json)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${PY:-python}"
NQ="${NQ:-1000}"
SCOPES="${SCOPES:-distractor cross}"
NPROBE="${NPROBE:-32}"
REF_RUN="${REF_RUN:-sweep_exports/20260826_223418_mmr-capacitor_centroid-nvm}"
LOGDIR="$ROOT/logs"; mkdir -p "$LOGDIR"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- preflight --
say "preflight"
for d in processed_hotpot_q1000 offline_data_hotpot_q1000 h_aimdc_sim; do
  [ -d "$d" ] || { echo "MISSING: $d/" >&2; exit 1; }
done
for f in h_aimdc_sim/fefet_ref_fit_smu.json h_aimdc_sim/igzo_fit.json h_aimdc_sim/igzo_fit.k.json; do
  [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
NQ_HAVE=$("$PY" -c "import json;print(len(json.load(open('processed_hotpot_q1000/hotpot_queries.json'))))")
[ "$NQ_HAVE" = "1000" ] || { echo "processed_hotpot_q1000 holds $NQ_HAVE queries, expected 1000" >&2; exit 1; }
"$PY" -c "import online_accelerator_device as m; assert hasattr(m, '_ws_count'), 'instrumentation missing'"
echo "ok: q1000 build, fits, instrumented pipeline"

say "unit test"
"$PY" -m pytest -q tests/test_write_stats.py 2>/dev/null || "$PY" tests/test_write_stats.py

# ------------------------------------------------ scratch workspace (q1000) --
WS="$ROOT/ws_q1000"
mkdir -p "$WS"
ln -sfn "$ROOT/processed_hotpot_q1000"   "$WS/processed_hotpot"
ln -sfn "$ROOT/offline_data_hotpot_q1000" "$WS/offline_data_hotpot"
ln -sfn "$ROOT/h_aimdc_sim"              "$WS/h_aimdc_sim"
[ -d "$ROOT/noise_configs" ] && ln -sfn "$ROOT/noise_configs" "$WS/noise_configs"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_DIRS=()
for SCOPE in $SCOPES; do
  OUT_BASE="$ROOT/sweep_exports/write_stats/$SCOPE"
  mkdir -p "$OUT_BASE"
  EXTRA=()
  [ "$SCOPE" = "cross" ] && EXTRA=(--nprobe "$NPROBE")

  say "retrieval: scope=$SCOPE  n=$NQ  (q1000 build)"
  # --output_dir is both the offline_data_<domain> base and the sweep_exports
  # base in sweep_eval.py, so run with "." inside the scratch workspace and
  # move the finished run directory afterwards.
  mkdir -p "$WS/sweep_exports"
  BEFORE="$(mktemp)"; ls "$WS/sweep_exports" | sort > "$BEFORE"
  ( cd "$WS" && "$PY" "$ROOT/sweep_eval.py" \
      --domain hotpot --num_queries "$NQ" \
      --retrieval_scope "$SCOPE" "${EXTRA[@]}" \
      --lambdas 0.7 --gammas 0.7 --tau_dups=-1.0 --top_ks 5 \
      --centroid_mode nvm \
      --centroid_fit_path h_aimdc_sim/fefet_ref_fit_smu.json \
      --centroid_reduce linear_sum --centroid_v_ml 1.4 \
      --mmr_mode capacitor --device_v_min -3.0 --device_v_max 3.0 \
      --output_dir . \
      2>&1 | tee "$LOGDIR/write_stats_${SCOPE}.log" )
  RUN_ID="$(ls "$WS/sweep_exports" | sort | comm -13 "$BEFORE" - | tail -1)"
  rm -f "$BEFORE"
  [ -n "$RUN_ID" ] || { echo "could not identify the new run_id for $SCOPE" >&2; exit 1; }
  mv "$WS/sweep_exports/$RUN_ID" "$OUT_BASE/"
  RUN_DIR="$OUT_BASE/$RUN_ID"
  echo "run_id = $RUN_ID  ->  $RUN_DIR"

  cat > "$RUN_DIR/run_flags.txt" <<FLAGS
run_id=$RUN_ID
generated_by=scripts/run_write_stats.sh
date=$(date -Iseconds)
purpose=IGZO-tier write/readout statistics (pre-write filter, writes per episode)
index_build=offline_data_hotpot_q1000 + processed_hotpot_q1000 (the n=1000 paper build, via ws_q1000/ symlinks)
retrieval_scope=$SCOPE
nprobe=$NPROBE $( [ "$SCOPE" = "cross" ] && echo "(active)" || echo "(irrelevant under distractor scope)" )
num_queries=$NQ
grid=lambda:0.7 tau_dup:-1.0 gamma:0.7 top_k:5
centroid_mode=nvm centroid_fit_path=h_aimdc_sim/fefet_ref_fit_smu.json centroid_reduce=linear_sum centroid_v_ml=1.4
mmr_mode=capacitor device_fit_path=h_aimdc_sim/igzo_fit.json device_scale_k_path=h_aimdc_sim/igzo_fit.k.json
device_v_min=-3.0 device_v_max=3.0 standardize_mmr_terms=True (default on)
FLAGS
  RUN_DIRS+=("$RUN_DIR")
done

# ------------------------------------------------------------------ analysis --
say "analysis"
for RD in "${RUN_DIRS[@]}"; do
  CHECK=()
  if [[ "$RD" == *"/distractor/"* && -d "$REF_RUN" && "$NQ" = "1000" ]]; then
    CHECK=(--check_against "$REF_RUN")
  fi
  "$PY" scripts/analyze_write_stats.py "$RD" "${CHECK[@]}"
done

say "done"
printf '%s\n' "${RUN_DIRS[@]}"
echo "summaries: <run_dir>/write_stats_summary.json"
