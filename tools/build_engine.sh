#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-yolov8n.pt}"
IMGSZ="${2:-640}"

# Ensure host-visible dirs exist (so the container can't write into a tmp overlay)
mkdir -p ros2_ws .ultralytics

echo "[build] docker compose build engine"
docker compose build engine

echo "[run] export ${MODEL} @ ${IMGSZ}px → ros2_ws/<model>.engine"
docker compose run --rm engine \
  python3 tools/export_engine.py "${MODEL}" "${IMGSZ}" --outdir ros2_ws

echo "[check] listing outputs:"
ls -lh ros2_ws/*.engine ros2_ws/*.json || true
