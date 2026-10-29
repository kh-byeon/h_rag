#!/usr/bin/env bash
# =============================================================================
# Four-arm simulation ablation on the n = 1,000 HotpotQA distractor set.
#
#   arm 1  digital L2                 : the L2 arm, identical in every run below
#   arm 2  digital MMR                : --mmr_mode software  (numpy L2 pair distances)
#   arm 3  calibrated average device  : --mmr_mode capacitor (the reported configuration)
#   arm 4  device with variation      : arm 3 + frozen per-cell variation of the DeCAP
#                                       coupling ratio, gain and threshold offset
#                                       (noise_configs/ablation/decap_<scenario>_s<seed>.json)
#
# The re-ranker, its parameters and the query set are those of scripts/run_q1000_distractor.sh
# (lambda 0.7, tau_dup -1.0, gamma 0.7, k 5). Retrieval only; generation is optional (GEN=1).
#
# Run from the repo root inside WSL with the project venv active:
#
#     bash scripts/run_ablation_q1000.sh
#     SCENARIOS="all_1x" SEEDS="1 2 3" bash scripts/run_ablation_q1000.sh   # a smaller grid
#     GEN=1 bash scripts/run_ablation_q1000.sh                             # + int8 generation
#
# Re-running skips runs already listed in ablation/manifest.tsv.
# Then:  python scripts/analyze_ablation.py ablation/manifest.tsv --out ablation/ablation_q1000
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${PY:-python}"
SCENARIOS="${SCENARIOS:-all_0.5x all_1x all_2x alpha_1x gain_1x offset_1x}"
SEEDS="${SEEDS:-1 2 3 4 5}"
GEN="${GEN:-0}"
OUT="$ROOT/ablation"; MAN="$OUT/manifest.tsv"; LOGDIR="$ROOT/logs"
mkdir -p "$OUT" "$LOGDIR" cache

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
[ -f "$MAN" ] || printf 'arm\tscenario\tseed\trun_id\n' > "$MAN"
done_already() { awk -F'\t' -v a="$1" -v s="$2" -v d="$3" '$1==a && $2==s && $3==d {f=1} END {exit !f}' "$MAN"; }

COMMON=(--domain hotpot --num_queries 1000 --retrieval_scope distractor
        --lambdas 0.7 --gammas 0.7 --tau_dups=-1.0 --top_ks 5)
DEVICE=(--centroid_mode nvm --centroid_fit_path h_aimdc_sim/fefet_ref_fit_smu.json
        --centroid_reduce linear_sum --centroid_v_ml 1.4
        --mmr_mode capacitor --device_v_min -3.0 --device_v_max 3.0)

run_one() {   # arm scenario seed  extra-args...
  local arm="$1" scen="$2" seed="$3"; shift 3
  if done_already "$arm" "$scen" "$seed"; then say "skip $arm/$scen/s$seed (in manifest)"; return; fi
  say "retrieval: $arm / $scen / seed $seed"
  local before; before="$(mktemp)"; ls sweep_exports 2>/dev/null | sort > "$before" || true
  "$PY" sweep_eval.py "${COMMON[@]}" "$@" 2>&1 | tee "$LOGDIR/ablation_${arm}_${scen}_s${seed}.log"
  local rid; rid="$(ls sweep_exports | sort | comm -13 "$before" - | tail -1)"; rm -f "$before"
  [ -n "$rid" ] || { echo "could not identify run_id" >&2; exit 1; }
  printf '%s\t%s\t%s\t%s\n' "$arm" "$scen" "$seed" "$rid" >> "$MAN"
  printf 'arm=%s\nscenario=%s\nseed=%s\nargs=%s\n' "$arm" "$scen" "$seed" "$*" > "sweep_exports/$rid/ablation_flags.txt"
}

say "variation configs"
"$PY" scripts/make_ablation_noise_configs.py --seeds "$(echo $SEEDS | tr ' ' ',')"

# arm 2: digital MMR (centroids digital too; under the distractor scope they do not act)
run_one digital_mmr none 0 --centroid_mode software --mmr_mode software
# arm 3: calibrated average device, re-run with this code version
run_one device_avg none 0 "${DEVICE[@]}"
# arm 4: frozen variation, one run per scenario and die (seed)
for scen in $SCENARIOS; do
  for s in $SEEDS; do
    run_one device_var "$scen" "$s" "${DEVICE[@]}" --noise_config "noise_configs/ablation/decap_${scen}_s${s}.json"
  done
done

if [ "$GEN" = "1" ]; then
  say "int8 generation for arms 2, 3 and arm 4 at all_1x"
  awk -F'\t' 'NR>1 && ($1!="device_var" || $2=="all_1x") {print $4}' "$MAN" | while read -r rid; do
    o="llm_out_ablation_int8/$rid"
    [ -n "$(ls "$o"/combos/llm_combo_*.json 2>/dev/null || true)" ] && continue
    "$PY" scripts/run_generation_sweep.py --retrieval_jsonl "sweep_exports/$rid/retrieval_$rid.jsonl" \
      --llm_output_root "$o" --quantization int8 --cache cache/gen_llama3-8b_int8.sqlite --chunk 1 \
      2>&1 | tee "$LOGDIR/ablation_gen_$rid.log"
  done
fi

say "done -- manifest: $MAN"
echo "next: $PY scripts/analyze_ablation.py $MAN --out ablation/ablation_q1000"
