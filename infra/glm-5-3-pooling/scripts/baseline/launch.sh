#!/bin/bash
# Host-side launcher. Does not start containers or stop existing services.
set -euo pipefail
ROLE=${1:?usage: launch.sh p|d|proxy|meta}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=$(basename "$HERE")
case "$ROLE" in p) PORT=8100;; d) PORT=8200;; proxy) PORT=9000;; meta) PORT=5000;; *) exit 2;; esac
[[ "$MODE" != baseline || "$ROLE" != meta ]] || { echo 'Baseline does not use Meta'; exit 2; }
[[ "$(docker inspect -f '{{.State.Running}}' vllm-pool)" == true ]] || { echo 'vllm-pool is not running; inspect container before starting it'; exit 2; }
if ss -H -lnt | awk '{print $4}' | grep -Eq ":${PORT}$"; then
  echo "Port $PORT is already in use; refusing duplicate launch"; exit 2
fi
BASE=/home/d00937186/glm53-pooling-lab
mkdir -p "$BASE/log/services/$MODE" "/run/lock/glm53-pooling-lab"
LOGDIR=$(mktemp -d "$BASE/log/services/$MODE/${ROLE}-$(date +%Y%m%d-%H%M%S)-XXXXXX")
nohup flock -n "/run/lock/glm53-pooling-lab/$ROLE.lock" bash "$HERE/run-container.sh" "$ROLE" "$LOGDIR" > "$LOGDIR/$ROLE.log" 2>&1 < /dev/null &
LAUNCHER_PID=$!
printf 'Launcher PID: %s\nLog: %s\n' "$LAUNCHER_PID" "$LOGDIR/$ROLE.log"
printf 'tail -F %q\n' "$LOGDIR/$ROLE.log"
