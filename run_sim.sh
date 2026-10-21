#!/usr/bin/env bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
set -euo pipefail

WORKDIR="${1:-.}"
cd "${WORKDIR}"

if [[ ! -f sim_igzo.sp ]]; then
  echo "[run_sim] sim_igzo.sp not found in ${WORKDIR}" >&2
  exit 1
fi
if [[ ! -f igzo_tft.osdi ]]; then
  echo "[run_sim] igzo_tft.osdi not found in ${WORKDIR}" >&2
  echo "[run_sim] Compile the Verilog-A model first." >&2
  exit 1
fi
if [[ ! -f igzo_measure.csv ]]; then
  echo "[run_sim] igzo_measure.csv not found in ${WORKDIR}" >&2
  exit 1
fi

echo "[run_sim] Running Xyce on sim_igzo.sp ..."
if command -v mpiexec >/dev/null 2>&1; then
  mpiexec -np 1 Xyce sim_igzo.sp
else
  Xyce sim_igzo.sp
fi

if [[ -f sim_igzo.prn ]]; then
  SIM_OUT="sim_igzo.prn"
elif [[ -f sim_igzo.csv ]]; then
  SIM_OUT="sim_igzo.csv"
else
  echo "[run_sim] No Xyce output file found (expected sim_igzo.prn or sim_igzo.csv)." >&2
  exit 1
fi

echo "[run_sim] Plotting simulation vs measured data ..."
python3 plot_comparison.py --sim "${SIM_OUT}" --meas igzo_measure.csv --out iv_comparison.png
echo "[run_sim] done. Saved iv_comparison.png"
