#!/bin/bash
# Relocated frozen PCP baseline; see ../../handoff-pcp.md.
unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY
source /root/.bashrc
WORKDIR=/home/d00937186/glm53-pooling-lab
mkdir -p "$WORKDIR/log/services/baseline"
cd "$WORKDIR"
export ASCEND_PROCESS_LOG_PATH="$WORKDIR/log/services/baseline/ascend-p-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=300
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
# enable_dsa_cp and use_ascend_direct absent from installed source; not injected.
export VLLM_USE_FASTOKENS=1
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=180
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

export HCCL_ALGO=level0:fullmesh
export HCCL_IF_IP=192.168.45.69
export HCCL_SOCKET_IFNAME=enp34s0f1
export GLOO_SOCKET_IFNAME=enp34s0f1
export TP_SOCKET_IFNAME=enp34s0f1
unset HCCL_INTRA_ROCE_ENABLE

export PYTHONHASHSEED=0
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'


KV_TRANSFER='{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30100","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":8,"tp_size":1}}}'

echo "[$(basename $0)] $(date '+%F %T') launching vllm serve ..."
exec vllm serve /home/weights/GLM-5.3-w4a4c8-mxfp4 \
  --host 0.0.0.0 \
  --port 8100 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 133120 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 20 \
  --served-model-name glm \
  --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_flashcomm1":true}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/d00937186/glm53-pooling-lab/profiling/p","torch_profiler_with_stack":false}' \
  --kv-transfer-config "$KV_TRANSFER"
