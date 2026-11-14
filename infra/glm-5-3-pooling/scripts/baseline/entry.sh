#!/bin/bash
set -euo pipefail
ROLE=${1:?usage: entry.sh p|d|proxy|meta}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ "$ROLE" =~ ^(p|d|proxy)$ ]] || exit 2
unset MMC_LOCAL_CONFIG_PATH
exec bash "$HERE/start-$ROLE.sh"
