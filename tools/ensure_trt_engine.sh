#!/bin/bash
set -euo pipefail

ENGINE_PATH="${YOLO_ENGINE:-/work/ros2_ws/yolov8n.engine}"
ONNX_PATH="${YOLO_ONNX:-/work/ros2_ws/yolov8n.onnx}"
LOGGER_TAG="ensure_trt_engine"

log() {
  echo "[$LOGGER_TAG] $*" >&2
}

if [ ! -x /usr/src/tensorrt/bin/trtexec ]; then
  log "trtexec missing; ensure TensorRT is installed. Skipping engine build."
else
  build_needed=false
  if [ ! -f "$ENGINE_PATH" ]; then
    log "No TensorRT engine at $ENGINE_PATH"
    build_needed=true
  elif [ "$ONNX_PATH" -nt "$ENGINE_PATH" ]; then
    log "ONNX model newer than engine; regenerating."
    build_needed=true
  fi

  if [ "$build_needed" = true ]; then
    mkdir -p "$(dirname "$ENGINE_PATH")"
    log "Building TensorRT engine from $ONNX_PATH"
    /usr/src/tensorrt/bin/trtexec \
      --onnx="$ONNX_PATH" \
      --saveEngine="$ENGINE_PATH" \
      --memPoolSize=workspace:2048M \
      --fp16 \
      --skipInference
    log "Engine written to $ENGINE_PATH"
  else
    log "Existing TensorRT engine is up to date."
  fi
fi

if [ "${1:-}" = "--check" ]; then
  exit 0
fi

exec "$@"
