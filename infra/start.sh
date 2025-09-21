#!/usr/bin/env bash
set -eo pipefail

# Relax nounset while sourcing ROS (prevents AMENT_TRACE_SETUP_FILES issue)
: "${ROS_DISTRO:=humble}"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

# Config
: "${FOXGLOVE_PORT:=8765}"
: "${FOXGLOVE_ADDRESS:=0.0.0.0}"          # bind address
: "${ENABLE_ROSBRIDGE:=false}"            # "true" to also run rosbridge_websocket (9090)
: "${ROSBRIDGE_PORT:=9090}"

echo "[infra] launching foxglove_bridge ws://${FOXGLOVE_ADDRESS}:${FOXGLOVE_PORT}"
# Start foxglove_bridge (foreground)
ros2 run foxglove_bridge foxglove_bridge \
  --ros-args \
  --params-file /work/params/foxglove.params.yaml \
  -p port:=${FOXGLOVE_PORT} \
  -p address:=${FOXGLOVE_ADDRESS} &
FG_PID=$!

# Optionally launch rosbridge websocket (useful for quick JSON tooling)
if [[ "${ENABLE_ROSBRIDGE,,}" == "true" ]]; then
  echo "[infra] launching rosbridge_websocket on :${ROSBRIDGE_PORT}"
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
    port:=${ROSBRIDGE_PORT} &
  RB_PID=$!
fi

# Graceful shutdown
_term() {
  echo "[infra] stopping..."
  kill ${FG_PID} >/dev/null 2>&1 || true
  if [[ -n "${RB_PID:-}" ]]; then kill ${RB_PID} >/dev/null 2>&1 || true; fi
  wait || true
}
trap _term INT TERM

# Wait on foxglove_bridge
wait ${FG_PID}
