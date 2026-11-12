---
title: "GLM-5.3 双机UB三级池化部署"
type: implementation
domain: infra
status: active
---

# GLM-5.3 W4A4C8 · 双机 16 卡 UB FM16 · 三级池化部署复现手册

> **现场**：`192.168.45.69` / `192.168.45.77`，各 8×Ascend950DT，UB FM16，CANN 9.2.1，

> 镜像 `quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc-a5-openeuler`，权重 `<权重路径>`（例：`/home/weights/GLM-5.3-w4a4c8-mxfp4`，两台按实际填写）。
> **工作目录**：两台均为 `/home/d00937186`（脚本及日志路径）。

---

## 1. 容器拉起

### 1.1 前置条件

| 检查项 | 命令 | 期望 |
| --- | --- | --- |
| 镜像已在本地 | `docker images \| grep vllm-ascend:nightly-releases-v0.32.0rc-a5` | 有该 tag（≈32.4 GB） |
| 卡在位 | `npu-smi info \| grep -c Ascend950DT` | `8` |
| 权重可见 | `ls -d <权重路径>` | 存在（391 GB / 116 项；两台路径可不同，脚本里各自填） |

### 1.2 拉起容器参考脚本：`start_container.sh`

```bash
#!/bin/bash
NAME=vllm-pool                        # ← 占位符 1：容器名
IMAGE_ID=quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc-a5-openeuler   # ← 占位符 2：镜像 ID/tag

WORKDIR=/home/d00937186
# 本现场挂载本机 /etc/hccl_rootinfo.json；缺失曾导致 MoE AllToAll 建域报 error code 4。
# 挂载前检查 topo_file_path、rank/device/local_id、地址及端口与实际通信方式一致。
# 本次走 UB，使用 UB topo（atlas_850_2.json）；不能误用 UBOE topo。
# 后续走 UBOE 时，需同步核对 UBOE topo、设备地址、端口、协议及运行库支持，不能只换文件名。
# topo_file_path 引用文件也必须在容器内可读；勿跨机器复制 rootinfo。
# 下方沿用现场 rw 挂载，不代表只读一定不可用；容器内改写会影响宿主机及其他使用者。
docker run -u root -it -d --name "$NAME" \
  --net=host --privileged=true --ipc=host --shm-size=50g --ulimit memlock=-1:-1 \
  --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/ummu --device=/dev/uburma \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/bin/urma_admin:/usr/bin/urma_admin \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
  -v /etc/hixlep:/etc/hixlep \
  -v /var/log/npu:/usr/slog \
  -v /home:/home \
  -v /mnt:/mnt \
  -w "$WORKDIR" \
  "$IMAGE_ID" bash
```

---

## 2. 二级 MemCache（HBM → 池化 DRAM）配置

> 目标：**只在节点A** 起全集群唯一的 MetaService；两个节点都准备好 LocalService 配置。
> 二级 = **SSD 关闭**。LocalService 不单独起 —— 由第 3 步每个 vLLM worker 进程在**进程内**拉起，读的就是本节的 `mmc-local.conf`。
> 本节为**通用步骤**，地址、协议、容量和网卡需按实际环境填写。

### 2.0 约定

- `<节点A_IP>` = **节点A 的服务器 IP**（例 `192.168.45.69`），两台的 conf 都填这一个。
- `<容器名>` = 你启动的容器名；`/home/d00937186` = 容器内工作目录（与宿主机同路径的 bind mount）。

### 2.1 【宿主机】端口占用与服务检查

```bash
docker exec <容器名> bash -c "pgrep -af 'vllm serve|MetaService' | grep -v pgrep || echo 无进程"
ss -ltn | grep -E ':(5000|6000|8000) ' || echo "5000/6000/8000 空闲"
```

期望：**无 memcache 相关进程**、三个端口空闲。

> 容器用 `--net=host`，端口在宿主机和容器内是同一份，`ss` 在宿主机跑即可。

### 2.2 【节点A（容器内）】查询 memcache 配置路径

容器内执行：

```bash
# ① memcache 包目录（下称 <MC_PKG>）
MC_PKG=$(python3 -c "import memcache_hybrid, os; print(os.path.dirname(memcache_hybrid.__file__))")
echo "$MC_PKG"

# ② 配置模板就在 $MC_PKG/config 下
ls -l "$MC_PKG/config/"
```

两个模板文件：

| 文件 | 谁用 | 说明 |
| --- | --- | --- |
| `<MC_PKG>/config/mmc-meta.conf` | 节点A 的 MetaService | 元数据服务自身配置 |
| `<MC_PKG>/config/mmc-local.conf` | **每个 vLLM worker 进程**（进程内 LocalService） | 池容量与数据面协议 |

> ⚠️ 模板默认值是 `127.0.0.1` + `host_rdma` + `1GB`，**不能直接用**，需要按需修改。
> 建议将模板复制到工作目录后再修改（二级放 `/home/d00937186/mmc/conf/l2/` 下），后续用 `export MMC_META_CONFIG_PATH=/home/d00937186/mmc/conf/l2/mmc-meta.conf` 使配置生效
> （MetaService 用 `MMC_META_CONFIG_PATH`；vLLM 侧 LocalService 用 `MMC_LOCAL_CONFIG_PATH`，见第 3 步）。

### 2.3 【节点A（容器内）】修改元数据配置 `mmc-meta.conf`（仅主节点）

**配置模板：**

```ini
ock.mmc.meta_service_url = tcp://<节点A_IP>:5000
ock.mmc.meta_service.config_store_url = tcp://<节点A_IP>:6000
ock.mmc.meta_service.metrics_url = http://<节点A_IP>:8000
ock.mmc.log_level = info
ock.mmc.evict_threshold_high = 70
ock.mmc.evict_threshold_low = 60
```

**参数解析：**

| 参数 | 必填 | 取值 | 关注点 |
| --- | --- | --- | --- |
| `ock.mmc.meta_service_url` | 是 | `tcp://<节点A_IP>:5000` | **所有节点/所有角色必须同一个值**；写 IP 不要写 127.0.0.1 |
| `ock.mmc.meta_service.config_store_url` | 是 | `tcp://<节点A_IP>:6000` | 必须与 `mmc-local.conf` 里 `local_service.config_store_url` 一致 |
| `ock.mmc.meta_service.metrics_url` | 是 | `http://<节点A_IP>:8000` | 根据空闲端口填写 |
| `ock.mmc.evict_threshold_high/low` | 否 | `70` / `60` | L2 淘汰水位。开 SSD 时用 70/60 提高 SSD 命中；纯二级可留默认 90/80 |

### 2.4 【节点A、节点B（容器内）】修改 `mmc-local.conf`（两台内容相同，由 vLLM worker 进程读取）

**配置模板：**

```ini
ock.mmc.meta_service_url = tcp://<节点A_IP>:5000
ock.mmc.local_service.config_store_url = tcp://<节点A_IP>:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_urma
ock.mmc.local_service.dram.size = 50GB
ock.mmc.local_service.max.dram.size = 1024GB
# 二级：SSD 关闭（相关配置均不写，开启后再补充）
ock.mmc.local_service.storage.enabled = false
```

> 本环境实际取值：`world_size = 64`、`dram.size = 8GB`（每 worker 8GB → 每节点 8×8 = **64 GB**，两节点合计 **128 GB**）。上面的 `256` / `50GB` 只是示例，`max.dram.size` 可以留空（默认 1TB），不影响二级功能。

**关键参数：**

| 参数 | 必填 | 取值 | 关注点 |
| --- | --- | --- | --- |
| `ock.mmc.local_service.protocol` | 是 | `device_urma` | A5/950DT 走 UB 固定用这个；A3 HCCS 用 `device_sdma`，UBOE 用 `device_uboe`，A2 用 `device_rdma`。**默认 `host_rdma` 是错的** |
| `ock.mmc.local_service.dram.size` | 是 | `50GB` | **每个 worker 进程**贡献的 DRAM；`device_*` 协议要求 1GB 对齐。容量 = 本值 × 本节点进程数 |
| `ock.mmc.local_service.world_size` | 否 | `256` | 支持的 LocalService 上限，需 ≥ 全集群总进程数；**一旦有 rank 连上就不许再改**，改要重启 MetaService |
| `ock.mmc.local_service.max.dram.size` | 否 | `1024GB` | 各 rank 贡献不同大小时的封顶值，保持 ≥ `dram.size` |
| `ock.mmc.local_service.storage.enabled` | 否 | 二级 `false` | 只有开三级才改 `true`，并补齐 `ubsio.disk.path` / `ubsio.mem.size_in_gb` / `ubsio.standalone.device_count` / `ubsio.standalone.force_new_disk` |

### 2.5 【节点A（容器内）】启动 MetaService（后台 + 日志重定向）

```bash
# 环境：告诉 MetaService 读哪个配置；把 memcache 自带 lib 加进库路径
export MMC_META_CONFIG_PATH=/home/d00937186/mmc/conf/l2/mmc-meta.conf
MC_PKG=$(python3 -c "import memcache_hybrid, os; print(os.path.dirname(memcache_hybrid.__file__))")
export LD_LIBRARY_PATH=$MC_PKG/lib:$LD_LIBRARY_PATH
mkdir -p /home/d00937186/mmc/log

# 启动：nohup 保证退出容器后进程不被挂断；日志重定向到文件
nohup python3 -c "from memcache_hybrid import MetaService; MetaService.main()" \
  > /home/d00937186/mmc/log/pool.log 2>&1 &

```

查看日志：

```bash
tail -f /home/d00937186/mmc/log/pool.log                                    # 持续跟
tail -n 50 /home/d00937186/mmc/log/pool.log                                 # 只看最后 50 行
grep -inE 'error|fail|exception|refused' /home/d00937186/mmc/log/pool.log   # 只找问题
```

确认端口已监听：

```bash
ss -ltn | grep -E ':(5000|6000|8000) '
```

期望：`5000` / `6000` / `8000` 三个 LISTEN；`pool.log` 无 `ERROR`；`pgrep -af 'MetaService.main'` 能看到进程。

停止与确认：

```bash
pgrep -af MetaService.main                 # 核对本任务进程
kill -TERM <已确认的MetaService_PID>       # 容器内执行，PID须来自同一命名空间
pgrep -af MetaService.main                 # 确认已退出
ss -ltn | grep -E ':(5000|6000|8000) ' || echo 端口已释放                # 端口全释放才算停干净
```

> 停 MetaService 前先停 vLLM；重启时顺序反过来（MetaService → vLLM）。
> MemCache 自己的日志在容器内 `/var/log/memcache_hybrid/logs/mmc-meta.log`，连接类问题查这里。

### 2.6 【节点B（容器内）】验证到节点A 的连通性

在节点B 的容器内（容器是 `--net=host`，与宿主机同网络栈）执行：

```bash
for p in 5000 6000 8000; do
  timeout 2 bash -c "</dev/tcp/<节点A_IP>/$p" 2>/dev/null && echo "$p OK" || echo "$p FAIL"
done
```

期望：三个 `OK`。节点A 上也跑一遍同样命令（应为全 OK）。出现 `FAIL` → 依次查：节点A 的 `ss -ltn`、`meta_service_url` 是否写错 IP、防火墙。

### 2.7 容量口径

> ⚠️ MetaService 单独起来**不会**让池容量出现 —— 容量由每个 vLLM worker 进程各自的 LocalService 注册进池后才形成。此时"没容量"是正常的，不是配错。等待vllm服务拉起后再查看即可

## 3. PD 服务接入 memcache 后端并拉起（二级）

> 测试case：**GLM-5.3 W4A4C8**、950DT、16 卡 = **1P（TP8）+ 1D（TP8）各一台**。
> P 用 `8100`、D 用 `8200`、proxy 用 `9000`（不要用 8000，节点A 的 MetaService metrics 已占）。

### 3.0 约定

- `192.168.45.69` / `192.168.45.77`：P / D 节点业务网 IP；`enp34s0f1`：业务网卡名（两台同名）
- 工作目录 `/home/d00937186`（与宿主机 bind mount 同路径）；日志统一 `/home/d00937186/mmc/log/`
- `<权重路径>`：GLM-5.3-w4a4c8-mxfp4 目录，两台按实际填（NFS 或本地盘均可）

### 3.1 【P 节点（容器内）】Prefill 完整脚本 `start-p.sh`（本机 192.168.45.69）

`# ★` 开头的是 PD + 池化新增项；其余是 GLM-5.3 W4A4C8 / 950DT / TP8 的原始配置。

```bash
#!/bin/bash
# ============================================================================
# GLM-5.3 W4A4C8 · 950DT · PD 分离 —— P 节点（Prefill）
#   本机：192.168.45.69   8×Ascend950DT   TP8   kv_producer
#   用法：nohup bash start-p.sh > /home/d00937186/mmc/log/serve-p.log 2>&1 &
#   ★ = 为 PD + 池化新增的项；其余是 GLM-5.3 W4A4C8 单机 TP8 的原始配置
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
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_FASTOKENS=1
export VLLM_SERVER_DEV_MODE=1                 # 便于 /reset_prefix_cache 验证 L2/L3
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

# ---------- ★ PD 跨机 + 池化通信变量 ----------
export PYTHONHASHSEED=0
export HCCL_ALGO=level0:fullmesh
export HCCL_IF_IP=192.168.45.69
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

# ---------- ★ 池化后端：MultiConnector = P2D 传输 + memcache 池 ----------
KV_TRANSFER='{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30001","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"0","use_layerwise":false}}]}}'

echo "[$(basename $0)] $(date '+%F %T') launching vllm serve ..."
exec vllm serve <权重路径> \
  --host 0.0.0.0 \
  --port 8100 \
  --api-server-count 1 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 135000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 16 \
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
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
  --additional-config '{"enable_flashcomm1": true, "enable_cpu_binding": true, "enable_fused_mc2": 1}' \
  --kv-transfer-config "$KV_TRANSFER"
```

### 3.2 【D 节点（容器内）】Decode 完整脚本 `start-d.sh`（本机 192.168.45.77）

`# ★` 开头的是 PD + 池化新增项；与 P 的差异只有 **IP / 端口 / kv 角色 / MTP 投机**。

```bash
#!/bin/bash
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
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_FASTOKENS=1
export VLLM_SERVER_DEV_MODE=1                 # 便于 /reset_prefix_cache 验证 L2/L3
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
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
exec vllm serve <权重路径> \
  --host 0.0.0.0 \
  --port 8200 \
  --api-server-count 1 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 135000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 16 \
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
```


> **MTP配置**：本例P、D均开启MTP，P步长1、D步长5，步长不必相同。P不开MTP而D开启时，当前连接器会因缺少第78层KV元数据而失败。

### 3.3 【对应节点容器内】拉起服务（后台 + 日志重定向）

3.1 / 3.2 两份脚本已放在对应机器的 `/home/d00937186/vllm-scripts/` 下，直接拉起：

```bash
mkdir -p /home/d00937186/mmc/log
nohup bash /home/d00937186/vllm-scripts/start-pool.sh l2 p > /home/d00937186/mmc/log/serve-p.log 2>&1 &     # P 先起
nohup bash /home/d00937186/vllm-scripts/start-pool.sh l2 d > /home/d00937186/mmc/log/serve-d.log 2>&1 &     # D 后起
```

查看日志：

```bash
tail -f /home/d00937186/mmc/log/serve-p.log
grep -inE 'error|traceback|failed|refused' /home/d00937186/mmc/log/serve-p.log
```

### 3.4 【69容器内】拉起 PD proxy

PD 分离的请求必须经 proxy 分流到 P 和 D。脚本用仓库自带的示例：

```bash
# 该脚本已在 /home/d00937186/vllm-scripts/load_balance_proxy_server_example.py；若缺失再从仓库拷：
# cp <仓库路径>/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py /home/d00937186/vllm-scripts/

nohup python3 /home/d00937186/vllm-scripts/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 9000 \
  --prefiller-hosts 192.168.45.69 --prefiller-ports 8100 \
  --decoder-hosts 192.168.45.77 --decoder-ports 8200 \
  > /home/d00937186/mmc/log/proxy.log 2>&1 &

tail -f /home/d00937186/mmc/log/proxy.log
```


### 3.5 冒烟

```bash
curl -s http://192.168.45.69:9000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm","prompt":"Hello, the capital of France is","max_tokens":32,"temperature":0}'
```

判读：返回 `choices[0].text` 非空即可。同一个长前缀再发一次，命中情况需结合缓存指标判断，不能仅看响应速度（三级检查见 §4.6）。


## 4. 开启三级池化（HBM → DRAM → SSD）

> 沿用第3节的 P/D 脚本，三级配置单独放在 `mmc/conf/l3/`，不覆盖 `mmc/conf/l2/`。以下启动命令均在容器内执行。

### 4.0 约定

- 工作目录：`/home/d00937186`，两台同名但非共享。
- MetaService 仅在69启动；P在69、D在77，proxy在69。
- 切换前先停止本任务旧服务，启动顺序为 MetaService → P/D → proxy；已有对应服务时不要重复启动。

### 4.1 磁盘要求

- 使用专用、干净的 SSD 块设备或分区，无文件系统、无挂载、无其他业务占用，建议与业务数据分盘。
- 设备必须在容器内可见且可读写；`ubsio.disk.path` 填本机实际设备路径。
- **`ubsio.standalone.force_new_disk=true` 会在服务启动时强制初始化指定设备，清除原缓存数据。不得指向业务数据盘。**

### 4.2 【两台容器内】三级配置

在 `/home/d00937186/mmc/conf/l3/mmc-local.conf` 中保留二级的连接与 DRAM 配置，并启用以下参数：

```ini
ock.mmc.local_service.storage.enabled = true
ubsio.disk.path = /dev/nvme3n1p1
ubsio.mem.size_in_gb = 5
ubsio.standalone.device_count = 8
ubsio.standalone.force_new_disk = true
```

| 参数 | 说明 |
| --- | --- |
| `ubsio.disk.path` | 专用 SSD 设备或分区；示例为每台150 GiB分区，按实际填写 |
| `ubsio.mem.size_in_gb` | 每进程主机内存缓冲池，开启 SSD 至少5GB；不占用 SSD 分区容量 |
| `ubsio.standalone.device_count` | 本机启用 DRAM 的 LocalService 数量，本例为8 |
| `ubsio.standalone.force_new_disk` | `true` 表示启动时强制初始化缓存设备 |

> `mmc/conf/l3/mmc-meta.conf` 的连接地址须与 LocalService 一致。本现场服务端口为5000/6000，metrics为8000；各节点按同一份实际配置连接。

### 4.3 【69容器内】启动 MetaService

```bash
mkdir -p /home/d00937186/mmc/log
nohup bash /home/d00937186/vllm-scripts/start-meta.sh l3 \
  > /home/d00937186/mmc/log/l3-meta.log 2>&1 &
```

查看日志：

```bash
tail -f /home/d00937186/mmc/log/l3-meta.log
tail -f /var/log/memcache_hybrid/logs/mmc-meta.log
```

### 4.4 【P/D容器内】拉起模型服务

沿用 §3.1 / §3.2 的脚本：**P开启MTP步长1，D开启MTP步长5**。`start-pool.sh l3` 选择三级配置并挂载本机 UB 拓扑。

```bash
# 69：P节点
mkdir -p /home/d00937186/mmc/log
export CONFIRM_SSD_USE=/dev/nvme3n1p1  # 必须与本机 ubsio.disk.path 一致
nohup bash /home/d00937186/vllm-scripts/start-pool.sh l3 p \
  > /home/d00937186/mmc/log/l3-p.log 2>&1 &

# 77：D节点
mkdir -p /home/d00937186/mmc/log
export CONFIRM_SSD_USE=/dev/nvme3n1p1
nohup bash /home/d00937186/vllm-scripts/start-pool.sh l3 d \
  > /home/d00937186/mmc/log/l3-d.log 2>&1 &
```

查看日志（在对应节点执行）：

```bash
tail -f /home/d00937186/mmc/log/l3-p.log
tail -f /home/d00937186/mmc/log/l3-d.log
```

### 4.5 【69容器内】拉起 PD proxy

```bash
nohup bash /home/d00937186/vllm-scripts/start-proxy.sh \
  > /home/d00937186/mmc/log/l3-proxy.log 2>&1 &
```

查看日志：

```bash
tail -f /home/d00937186/mmc/log/l3-proxy.log
```

### 4.6 容量与推理检查

```bash
curl -f http://192.168.45.69:8100/health
curl -f http://192.168.45.77:8200/health
curl -fsS http://192.168.45.69:8000/metrics \
  | grep -E '^memcache_(total_capacity_bytes|allocated_bytes)\{'
```

推理请求沿用 §3.5。容量以字节计；池指标中的 HBM 不包含 vLLM 自己管理的 L1 KV Cache。

> 三级命中须有 SSD 写入、目标前缀从L2淘汰、清L1后重放触发 SSD 读取的日志或指标证据。仅请求成功或响应变快不能证明三级命中。

## 5. 后续测试

见 [测试计划与操作指南](test-plan.md)。

## 附录 · 三级池化四级介质

| 级 | 介质 | 归属 | 验证方式 |
| --- | --- | --- | --- |
| L1 | NPU HBM 前缀缓存 | 每个 vLLM 实例私有 | 同前缀请求，检查 vLLM 前缀缓存命中指标 |
| L2 | 池化 DRAM（MemCache LocalService） | 全集群一个池，各实例贡献 DRAM | `POST /reset_prefix_cache` 清 L1 后再请求仍命中 |
| L2.5 | UBS IO DRAM | 每进程一份 | `ubsio.mem.size_in_gb` 配置主机内存缓冲池 |
| L3 | 本地 SSD | 每节点 | 确认L2淘汰后清L1重放，检查SSD读取证据 |

> 一句话：**通信域可以有多个，池只有一个；L1 私有，L2/L3 共享。**

## Knowledge Extraction

- [ ] 待完成128K/1K及分介质命中验证后，复核可复用结论；本次仅归档部署与测试设计，不提前提炼未验证结论。

## 当前验证状态（2026-11-12）

现有容器已通过工作目录中的测试副本恢复P/D，统一max-num-seqs=16，关闭HBM前缀缓存，P1/D5不变。短请求外部池读写及128K/1K并发1、2已通过；后续已验证真实SSD读取及串行单窗口80/20，但并发2三轮性能验收尚未完成，详见[测试计划](test-plan.md#501-2026-11-12现场验证进展)。原始部署脚本保留，正式对比前需使用独立指标目录。
