#!/bin/bash
set -euo pipefail
W=/home/d00937186
LEVEL=${1:?usage: start-pool.sh l2|l3 p|d}
ROLE=${2:?usage: start-pool.sh l2|l3 p|d}
[[ "$LEVEL" =~ ^l[23]$ && "$ROLE" =~ ^[pd]$ ]] || exit 2
CONF="$W/mmc/conf/$LEVEL/mmc-local.conf"
test -s "$CONF"
test -s "$W/rootinfo/rootinfo-ub.json"
if [[ "$LEVEL" == l3 ]]; then
  DISK=$(sed -n 's/^ubsio.disk.path *= *//p' "$CONF" | tail -1)
  [[ "$DISK" == /dev/* && -b "$DISK" ]] || { echo "Edit l3/mmc-local.conf: select an approved dedicated disk/partition"; exit 2; }
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS "$DISK"
  [[ "${CONFIRM_SSD_USE:-}" == "$DISK" ]] || { echo "After reviewing disk ownership, set CONFIRM_SSD_USE=$DISK"; exit 2; }
  [[ -z "$(lsblk -nr -o MOUNTPOINTS "$DISK" | tr -d '[:space:]')" ]] || { echo "Mounted disk refused"; exit 2; }
fi
export MMC_LOCAL_CONFIG_PATH="$CONF"
unset TEST_MF_TRANSPORT_FULL_MESH TEST_FULL_MESH
# Private mount namespace: host and existing container mount are unchanged.
exec unshare --mount --propagation private bash -c '
set -e
mount --bind "$1" /etc/hccl_rootinfo.json
exec bash "$2"
' _ "$W/rootinfo/rootinfo-ub.json" "$W/vllm-scripts/start-$ROLE.sh"
