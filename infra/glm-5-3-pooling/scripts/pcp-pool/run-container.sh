#!/bin/bash
set -uo pipefail
ROLE=${1:?}
LOGDIR=${2:?}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
docker exec -e CONFIRM_SSD_USE="${CONFIRM_SSD_USE:-}" vllm-pool bash "$HERE/entry.sh" "$ROLE"
STATUS=$?
exit "$STATUS"
