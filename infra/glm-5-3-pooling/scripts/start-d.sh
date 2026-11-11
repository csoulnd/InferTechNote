#!/bin/bash
# PD MTP: P uses 1 speculative token, D uses 5; both provide the MTP cache layer.
# ============================================================================
# GLM-5.3 W4A4C8 · 950DT · PD 分离 —— D 节点（Decode）
#   本机：192.168.45.77   8×Ascend950DT   TP8   kv_consumer
#   用法：nohup bash start-d.sh > /home/d00937186/mmc/log/serve-d.log 2>&1 &
#   ★ = 为 PD + 池化新增的项；与 P 的差异只有 IP / port / kv 角色 / MTP
# ============================================================================
WORKDIR=/home/d00937186
mkdir -p "$WORKDIR/mmc/log"

# ---------- 基础环境 ----------
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_FASTOKENS=1
export VLLM_SERVER_DEV_MODE=1                 # 便于 /reset_prefix_cache 验证 L2/L3
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

# ---------- ★ PD 跨机 + 池化通信变量（IP 换成本机）----------
export PYTHONHASHSEED=0
export HCCL_ALGO=level0:fullmesh
export HCCL_IF_IP=192.168.45.77
export HCCL_SOCKET_IFNAME=enp34s0f1
export GLOO_SOCKET_IFNAME=enp34s0f1
unset HCCL_INTRA_ROCE_ENABLE ASCEND_GLOBAL_RESOURCE_CONFIG ASCEND_ENABLE_USE_FABRIC_MEM

# ---------- ★ MemCache 池化 + A5 UB 自动通信资源 ----------
export MMC_LOCAL_CONFIG_PATH="${MMC_LOCAL_CONFIG_PATH:-$WORKDIR/mmc/conf/l2/mmc-local.conf}"
export LD_LIBRARY_PATH=/usr/local/python3.12.16/lib/python3.12/site-packages/memcache_hybrid/lib:$LD_LIBRARY_PATH
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

# ---------- CANN / ATB ----------
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

# ---------- ★ 池化后端：D 角色 = kv_consumer（kv_port / lookup_rpc_port 与 P 不同）----------
KV_TRANSFER='{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"30002","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"1","use_layerwise":false}}]}}'

echo "[$(basename $0)] $(date '+%F %T') launching vllm serve ..."
exec vllm serve /home/weights/GLM-5.3-w4a4c8-mxfp4 \
  --host 0.0.0.0 \
  --port 8200 \
  --api-server-count 1 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 135000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 20 \
  --served-model-name glm \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --async-scheduling \
  --trust-remote-code \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}' \
  --additional-config '{"enable_flashcomm1": true, "enable_cpu_binding": true, "enable_fused_mc2": 1}' \
  --kv-transfer-config "$KV_TRANSFER"
