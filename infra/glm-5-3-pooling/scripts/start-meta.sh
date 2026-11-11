#!/bin/bash
set -euo pipefail
W=/home/d00937186
LEVEL=${1:?usage: start-meta.sh l2|l3}
[[ "$LEVEL" =~ ^l[23]$ ]] || exit 2
export MMC_META_CONFIG_PATH="$W/mmc/conf/$LEVEL/mmc-meta.conf"
test -s "$MMC_META_CONFIG_PATH"
export LD_LIBRARY_PATH=/usr/local/python3.12.16/lib/python3.12/site-packages/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}
exec python3 -c 'from memcache_hybrid import MetaService; MetaService.main()'
