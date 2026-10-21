#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-xyce-openvaf:7.8.0}"
WORKDIR_IN_CONTAINER="/work"

echo "[1/3] Building Docker image: ${IMAGE_NAME}"
docker build -f Dockerfile.xyce -t "${IMAGE_NAME}" .

echo "[2/3] Compiling Verilog-A with OpenVAF"
docker run --rm -v "$(pwd):${WORKDIR_IN_CONTAINER}" "${IMAGE_NAME}" \
  bash -lc "cd ${WORKDIR_IN_CONTAINER} && openvaf igzo_tft.va -o igzo_tft.so"

echo "[3/3] Running Xyce simulation"
docker run --rm -v "$(pwd):${WORKDIR_IN_CONTAINER}" "${IMAGE_NAME}" \
  bash -lc "cd ${WORKDIR_IN_CONTAINER} && Xyce -plugin ./igzo_tft.so sim_igzo.sp"

echo "[Done] Check sim_igzo.prn and log output."
