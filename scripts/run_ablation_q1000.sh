#!/usr/bin/env bash
# =============================================================================
# Simulation ablation on the n = 1,000 HotpotQA distractor set (reviewer request:
# digital L2, digital identical MMR, calibrated average-device MMR, variation-aware MMR).
#
#   digital_l2        : the L2 arm (taken from the digital_mmr_std run; it does not
#                       depend on the MMR backend)
#   digital_mmr_std   : --mmr_mode software_std  -- numpy L2 pair distances with the
#                       identical procedure of the device path (calibration pass,
#                       z-standardized tau_dup gate, spread-matched diversity term)
#   digital_mmr_plain : --mmr_mode software      -- numpy L2 without the per-query
#                       standardization (the software path as implemented; reference)
#   device_avg        : --mmr_mode capacitor     -- the reported configuration
#   device_var        : device_avg + frozen per-cell variation of the DeCAP coupling
#                       ratio, gain and threshold offset, one run per scenario and die
#                       (noise_configs/ablation/decap_<scenario>_s<seed>.json)
#
# Re-ranker parameters and query set are those of scripts/run_q1000_distractor.sh
# (lambda 0.7, tau_dup -1.0, gamma 0.7, k 5). Retrieval only; generation is optional (GEN=1).
#
# The working tree may hold another index build in processed_hotpot/ (the n = 3,500 run), so,
# as in run_write_stats.sh, the runs execute inside ws_q1000/, a directory of symlinks to the
# *_q1000 copies, and the preflight refuses to start unless that build holds 1,000 questions.
#
# Run from the repo root inside WSL with the project venv active:
#
#     bash scripts/run_ablation_q1000.sh
#     SCENARIOS="all_1x" SEEDS="1 2 3" bash scripts/run_ablation_q1000.sh   # a smaller grid
#     QEMB=sweep_exports/query_emb_q1000_searchtext.npz bash scripts/run_ablation_q1000.sh
#                                    # replay query embeddings instead of running the encoder
#     GEN=1 bash scripts/run_ablation_q1000.sh                             # + int8 generation
#
# Runs land in ablation/runs/<run_id>/; re-running skips runs listed in ablation/manifest.tsv.
# Then:  python scripts/analyze_ablation.py ablation/manifest.tsv --out ablation/ablation_q1000
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${PY:-python}"
SCENARIOS="${SCENARIOS:-all_0.5x all_1x all_2x alpha_1x gain_1x offset_1x}"
SEEDS="${SEEDS:-1 2 3 4 5}"
GEN="${GEN:-0}"
QEMB="${QEMB:-}"
NQ="${NQ:-1000}"
OUT="$ROOT/ablation"; MAN="$OUT/manifest.tsv"; RUNS="$OUT/runs"; LOGDIR="$ROOT/logs"
mkdir -p "$OUT" "$RUNS" "$LOGDIR"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- preflight --
say "preflight"
if [ -d "$ROOT/processed_hotpot_q1000" ] && [ -d "$ROOT/offline_data_hotpot_q1000" ]; then
  WS="$ROOT/ws_q1000"; mkdir -p "$WS"
  ln -sfn "$ROOT/processed_hotpot_q1000"   "$WS/processed_hotpot"
  ln -sfn "$ROOT/offline_data_hotpot_q1000" "$WS/offline_data_hotpot"
  ln -sfn "$ROOT/h_aimdc_sim"              "$WS/h_aimdc_sim"
  ln -sfn "$ROOT/noise_configs"            "$WS/noise_configs"
else
  WS="$ROOT"
fi
NQ_HAVE=$("$PY" -c "import json;print(len(json.load(open('$WS/processed_hotpot/hotpot_queries.json'))))")
if [ "$NQ" = "1000" ] && [ "$NQ_HAVE" != "1000" ]; then
  echo "$WS/processed_hotpot holds $NQ_HAVE questions, expected the n = 1,000 build" >&2; exit 1
fi
[ "$NQ_HAVE" -ge "$NQ" ] || { echo "only $NQ_HAVE questions available" >&2; exit 1; }
for f in h_aimdc_sim/fefet_ref_fit_smu.json h_aimdc_sim/igzo_fit.json h_aimdc_sim/igzo_fit.k.json; do
  [ -f "$ROOT/$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EXTRA_Q=()
[ -n "$QEMB" ] && EXTRA_Q=(--query_emb_map "$(cd "$(dirname "$QEMB")" && pwd)/$(basename "$QEMB")")
echo "ok: workspace $WS ($NQ_HAVE questions, running $NQ)${QEMB:+, query embeddings from $QEMB}"

[ -f "$MAN" ] || printf 'arm\tscenario\tseed\trun_id\n' > "$MAN"
done_already() { awk -F'\t' -v a="$1" -v s="$2" -v d="$3" '$1==a && $2==s && $3==d {f=1} END {exit !f}' "$MAN"; }

COMMON=(--domain hotpot --num_queries "$NQ" --retrieval_scope distractor
        --lambdas 0.7 --gammas 0.7 --tau_dups=-1.0 --top_ks 5 --output_dir .)
DEVICE=(--centroid_mode nvm --centroid_fit_path h_aimdc_sim/fefet_ref_fit_smu.json
        --centroid_reduce linear_sum --centroid_v_ml 1.4
        --mmr_mode capacitor --device_v_min -3.0 --device_v_max 3.0)

run_one() {   # arm scenario seed  extra-args...
  local arm="$1" scen="$2" seed="$3"; shift 3
  if done_already "$arm" "$scen" "$seed"; then say "skip $arm/$scen/s$seed (in manifest)"; return; fi
  say "retrieval: $arm / $scen / seed $seed"
  mkdir -p "$WS/sweep_exports"
  local before; before="$(mktemp)"; ls "$WS/sweep_exports" | sort > "$before"
  ( cd "$WS" && "$PY" "$ROOT/sweep_eval.py" "${COMMON[@]}" "${EXTRA_Q[@]}" "$@" ) \
    2>&1 | tee "$LOGDIR/ablation_${arm}_${scen}_s${seed}.log"
  local rid; rid="$(ls "$WS/sweep_exports" | sort | comm -13 "$before" - | tail -1)"; rm -f "$before"
  [ -n "$rid" ] || { echo "could not identify run_id" >&2; exit 1; }
  mv "$WS/sweep_exports/$rid" "$RUNS/"
  printf 'arm=%s\nscenario=%s\nseed=%s\nnum_queries=%s\nquery_emb_map=%s\nargs=%s\n' \
    "$arm" "$scen" "$seed" "$NQ" "${QEMB:-none (encoder)}" "$*" > "$RUNS/$rid/ablation_flags.txt"
  printf '%s\t%s\t%s\t%s\n' "$arm" "$scen" "$seed" "$rid" >> "$MAN"
  sleep 1   # run ids have one-second resolution
}

say "variation configs"
"$PY" scripts/make_ablation_noise_configs.py --seeds "$(echo $SEEDS | tr ' ' ',')"

# centroids are digital in the digital arms; under the distractor scope they do not act
run_one digital_mmr_std   none 0 --centroid_mode software --mmr_mode software_std
run_one digital_mmr_plain none 0 --centroid_mode software --mmr_mode software
run_one device_avg        none 0 "${DEVICE[@]}"
for scen in $SCENARIOS; do
  for s in $SEEDS; do
    run_one device_var "$scen" "$s" "${DEVICE[@]}" --noise_config "noise_configs/ablation/decap_${scen}_s${s}.json"
  done
done

if [ "$GEN" = "1" ]; then
  say "int8 generation for the reference arms and device_var at all_1x"
  awk -F'\t' 'NR>1 && ($1!="device_var" || $2=="all_1x") {print $4}' "$MAN" | while read -r rid; do
    o="llm_out_ablation_int8/$rid"
    [ -n "$(ls "$o"/combos/llm_combo_*.json 2>/dev/null || true)" ] && continue
    "$PY" scripts/run_generation_sweep.py --retrieval_jsonl "$RUNS/$rid/retrieval_$rid.jsonl" \
      --llm_output_root "$o" --quantization int8 --cache cache/gen_llama3-8b_int8.sqlite --chunk 1 \
      2>&1 | tee "$LOGDIR/ablation_gen_$rid.log"
  done
fi

say "done -- manifest: $MAN"
echo "next: $PY scripts/analyze_ablation.py $MAN --out ablation/ablation_q1000"
