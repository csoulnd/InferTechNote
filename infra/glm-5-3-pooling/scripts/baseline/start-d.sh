#!/bin/bash
# Relocated frozen PCP baseline; see ../../handoff-pcp.md.
source /root/.bashrc
unset ftp_proxy
unset https_proxy
unset http_proxy
WORKDIR=/home/d00937186/glm53-pooling-lab
mkdir -p "$WORKDIR/log/services/baseline"
cd "$WORKDIR"
export ASCEND_PROCESS_LOG_PATH="$WORKDIR/log/services/baseline/ascend-d-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1
# MRv2 EPLB is enabled with CLI flags below; legacy variable is not used.
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_USE_FASTOKENS=1
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=200
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

export HCCL_ALGO=level0:fullmesh
export HCCL_IF_IP=192.168.45.77
export HCCL_SOCKET_IFNAME=enp34s0f1
export GLOO_SOCKET_IFNAME=enp34s0f1
export TP_SOCKET_IFNAME=enp34s0f1
unset HCCL_INTRA_ROCE_ENABLE

export PYTHONHASHSEED=0
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'


KV_TRANSFER='{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"30300","engine_id":"3","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":8,"tp_size":1}}}'

echo "[$(basename $0)] $(date '+%F %T') launching vllm serve ..."
exec vllm serve /home/weights/GLM-5.3-w4a4c8-mxfp4 \
  --host 0.0.0.0 \
  --port 8200 \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"use_async":true,"communicator":"torch_gloo"}' \
  --max-model-len 133120 \
  --max-num-batched-tokens 240 \
  --max-num-seqs 60 \
  --served-model-name glm \
  --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --hf-overrides '{"use_index_cache":true,"index_topk_freq":4}' \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}' \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"finegrained_tp_config":{"lmhead_tensor_parallel_size":8}}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/d00937186/glm53-pooling-lab/profiling/d","torch_profiler_with_stack":false}' \
  --kv-transfer-config "$KV_TRANSFER"
