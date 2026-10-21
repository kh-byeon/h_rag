#!/usr/bin/env bash
# =============================================================================
# Anchor-protection probe -- RETRIEVAL ONLY. No LLM, no generation, ~1 min/run.
#
# Question: does exempting the top relevance-ranked candidates from the tau_dup
# dedup gate reduce the 54 queries where MMR drops gold that L2 kept, without
# collapsing the diversity that is the point of MMR?
#
#     bash scripts/run_anchor_protect_probe.sh
#
# Every flag matches the published n=1000 run (20260826_223418) exactly; only
# --anchor_protect_n differs, so the L2 arm must come out identical across runs.
# The comparator checks that and warns if it does not.
#
# Decision rule, decided BEFORE looking (so this is not a fishing trip):
#   * L2-only losses fall AND mmr_ild / mmr_breadth hold up  -> worth pursuing
#   * nothing changes at all                                 -> the gate never
#     fires at tau=-1.0; the losses come from the MMR score's diversity term,
#     not the dedup gate, and this whole idea is the wrong lever
#   * losses fall but diversity collapses                    -> this is just
#     gamma in disguise; reject it
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
BASELINE="${BASELINE:-sweep_exports/20260826_223418_mmr-capacitor_centroid-nvm}"
PROTECT_VALUES="${PROTECT_VALUES:-1 2}"
mkdir -p logs

[ -d "$BASELINE" ] || { echo "baseline run not found: $BASELINE" >&2; exit 1; }

ARGS=(--domain hotpot --num_queries 1000
      --retrieval_scope distractor
      --lambdas 0.7 --gammas 0.7 --tau_dups=-1.0 --top_ks 5
      --centroid_mode nvm
      --centroid_fit_path h_aimdc_sim/fefet_ref_fit_smu.json
      --centroid_reduce linear_sum --centroid_v_ml 1.4
      --mmr_mode capacitor --device_v_min -3.0 --device_v_max 3.0)

COMPARE=("$BASELINE:ap0")

for N in $PROTECT_VALUES; do
  printf '\n\033[1m==> retrieval sweep, anchor_protect_n=%s\033[0m\n' "$N"
  BEFORE="$(mktemp)"; ls sweep_exports | sort > "$BEFORE"
  "$PY" sweep_eval.py "${ARGS[@]}" --anchor_protect_n "$N" \
    2>&1 | tee "logs/anchor_protect_n${N}.log"
  RID="$(ls sweep_exports | sort | comm -13 "$BEFORE" - | tail -1)"
  rm -f "$BEFORE"
  [ -n "$RID" ] || { echo "could not identify the new run_id" >&2; exit 1; }
  echo "$N -> $RID"
  # sweep_eval's manifest records neither this flag nor the FeFET paths.
  printf 'anchor_protect_n=%s\nbaseline=%s\nflags=%s\n' \
    "$N" "$BASELINE" "${ARGS[*]}" > "sweep_exports/$RID/run_flags.txt"
  COMPARE+=("sweep_exports/$RID:ap$N")
done

printf '\n\033[1m==> comparison\033[0m\n'
"$PY" scripts/compare_anchor_protect.py "${COMPARE[@]}" | tee logs/anchor_protect_compare.log
