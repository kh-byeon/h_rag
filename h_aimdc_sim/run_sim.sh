#!/usr/bin/env bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# Compile igzo_tft.va with OpenVAF, then run Xyce on sim_igzo.sp.
# Usage: run_sim.sh [WORKDIR]   (default: /work)
set -euo pipefail

WORKDIR="${1:-/work}"
cd "${WORKDIR}"

if [[ ! -f igzo_tft.va ]]; then
  echo "[run_sim] igzo_tft.va not found in ${WORKDIR}" >&2
  exit 1
fi
if [[ ! -f sim_igzo.sp ]]; then
  echo "[run_sim] sim_igzo.sp not found in ${WORKDIR}" >&2
  exit 1
fi

echo "[run_sim] OpenVAF: compiling igzo_tft.va ..."
openvaf igzo_tft.va

echo "[run_sim] Xyce: running sim_igzo.sp ..."
if command -v mpiexec >/dev/null 2>&1; then
  mpiexec -np 1 Xyce sim_igzo.sp
else
  Xyce sim_igzo.sp
fi

echo "[run_sim] done."
