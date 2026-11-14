#!/bin/bash
set -euo pipefail
ROLE=${1:?usage: entry.sh p|d|proxy|meta}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$ROLE" in
  meta|proxy) exec bash "$HERE/start-$ROLE.sh";;
  p|d) ;;
  *) exit 2;;
esac
ENTRY="$HERE/start-$ROLE.sh"
W=/home/d00937186
CONF="$W/mmc/conf/l3/mmc-local.conf"
ROOTINFO="$W/rootinfo/rootinfo-ub.json"
test -s "$CONF"
test -s "$ROOTINFO"
DISK=$(sed -n 's/^ubsio.disk.path *= *//p' "$CONF" | tail -1)
[[ "$DISK" == /dev/nvme3n1p1 && -b "$DISK" ]] || { echo "Unexpected pool disk"; exit 2; }
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS "$DISK"
[[ "${CONFIRM_SSD_USE:-}" == "$DISK" ]] || { echo "Review ownership; force_new_disk may initialize $DISK. Set CONFIRM_SSD_USE."; exit 2; }
[[ -z "$(lsblk -nr -o MOUNTPOINTS "$DISK" | tr -d '[:space:]')" ]] || { echo "Mounted disk refused"; exit 2; }
export MMC_LOCAL_CONFIG_PATH="$CONF"
unset TEST_MF_TRANSPORT_FULL_MESH TEST_FULL_MESH
exec unshare --mount --propagation private bash -c '
set -e
mount --bind "$1" /etc/hccl_rootinfo.json
exec bash "$2"
' _ "$ROOTINFO" "$ENTRY"
