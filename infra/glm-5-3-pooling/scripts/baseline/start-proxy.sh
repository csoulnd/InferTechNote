#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/load_balance_proxy_server_example.py" --host 0.0.0.0 --port 9000 --prefiller-hosts 192.168.45.69 --prefiller-ports 8100 --decoder-hosts 192.168.45.77 --decoder-ports 8200
