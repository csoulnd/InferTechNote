#!/bin/bash
set -euo pipefail
exec python3 /home/d00937186/vllm-scripts/load_balance_proxy_server_example.py --host 0.0.0.0 --port 9000 --prefiller-hosts 192.168.45.69 --prefiller-ports 8100 --decoder-hosts 192.168.45.77 --decoder-ports 8200
