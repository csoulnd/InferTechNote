---
title: "GLM-5.3 双机UB池化部署"
type: implementation
domain: infra
status: active
---

# GLM-5.3 双机UB池化部署

本指南对应已归档的PCP8/DP8部署脚本，替代旧TP8/TP8示例。2026-11-14只读核验两机vllm-pool容器挂载；当前运行进程可能属于其他工作目录，不能仅凭端口或容器名停止服务。复用现有容器，不自动重建。模型与依赖安装不在本文范围内。

## 角色、目录与配置

| 内容 | 现场值 |
|---|---|
| P / Meta / proxy | 192.168.45.69 |
| D | 192.168.45.77 |
| 容器 | 两机各自vllm-pool |
| 工程 | /home/d00937186/glm53-pooling-lab，两机同路径但不共享存储 |
| 权重 | /home/weights/GLM-5.3-w4a4c8-mxfp4 |
| MMC配置 | /home/d00937186/mmc/conf/l3/mmc-local.conf、mmc-meta.conf |
| 各机rootinfo | /home/d00937186/rootinfo/rootinfo-ub.json，禁止跨机覆盖 |
| 服务日志 | 工程log/services/<模式>/<角色-启动批次>/ |

脚本入口：[baseline](scripts/baseline/launch.sh)、[pcp-pool](scripts/pcp-pool/launch.sh)。参数以同目录start-p.sh、start-d.sh为准。外层配置是唯一可编辑来源，工程内不保留第二份副本。

## 容器挂载与通信拓扑

以下是本次两机docker inspect核验的现场，不是适用于所有硬件的通用容器模板。

| 宿主源路径 / 设置 | 容器目的路径 / 作用 |
|---|---|
| /home，bind传播rslave | /home；权重、工程、外层配置及测试环境保持相同路径 |
| /mnt | /mnt；本现场其他挂载入口，未证明为本任务必需 |
| /usr/local/Ascend/driver | 同路径；驱动库及topo文件 |
| /usr/local/dcmi | 同路径；设备管理接口 |
| /usr/local/sbin/npu-smi、/usr/local/sbin | 同路径；设备诊断工具 |
| /usr/bin/urma_admin | 同路径；URMA诊断工具 |
| /etc/hccn.conf、/etc/hixlep | 同路径；通信配置 |
| /etc/hccl_rootinfo.json | 同路径；容器默认rootinfo入口 |
| /var/log/npu | /usr/slog；Ascend日志挂载 |
| network=host、ipc=host | 使用宿主网络和IPC，端口冲突在宿主检查 |
| privileged=true、memlock=-1、shm-size=50GiB | 本现场已配置；不是对其他环境的最低权限结论 |
| /dev/davinci0..7、davinci_manager、hisi_hdc、ummu、uburma | 当前8卡及UB设备映射 |

宿主机只读检查：

```bash
docker inspect vllm-pool --format '{{json .Mounts}}'
docker inspect vllm-pool --format '{{json .HostConfig.Devices}}'
docker inspect vllm-pool --format '{{.HostConfig.NetworkMode}} {{.HostConfig.IpcMode}} {{.HostConfig.Privileged}}'
docker exec vllm-pool test -r /home/d00937186/rootinfo/rootinfo-ub.json
docker exec vllm-pool test -r /usr/local/Ascend/driver/topo/950/atlas_850_2.json
```

两机本任务rootinfo的topo_file_path均指向 `/usr/local/Ascend/driver/topo/950/atlas_850_2.json`。rootinfo与其引用的topo是两个文件：只挂rootinfo而容器看不到topo仍不完整。应核对协议、设备/rank/local_id、地址、端口与本机实际拓扑；文件存在不等于通信建域成功。

池化entry.sh先检查本机权威rootinfo，再通过 `unshare --mount --propagation private` 创建私有挂载命名空间，将它bind到 `/etc/hccl_rootinfo.json`，随后exec模型服务。该覆盖只对该进程及子进程可见，不改写宿主rootinfo，也不代表普通docker exec看到同一覆盖。排查时以目标进程的mountinfo及其命名空间内可见文件为准。启动入口需要已有容器允许mount/unshare，本现场privileged满足此方式；不要直接改写宿主的默认rootinfo文件。

本任务使用UB/device_urma；UBOE需要重新匹配协议、topo、地址和运行库，不能只替换文件名。`HCCL_ALGO=level0:fullmesh`是HCCL算法配置，与实验性的 `TEST_MF_TRANSPORT_FULL_MESH` / `TEST_FULL_MESH` 开关不同；当前入口显式取消后两者。

基线entry.sh与池化入口并非同一拓扑包装，复现前分别检查。基线依赖容器默认通信配置也必须有效；不要假设它自动使用池化入口的私有rootinfo挂载。

## 模型与连接方式

| 参数 | P | D |
|---|---|---|
| HTTP端口 | 8100 | 8200 |
| DP / TP | 1 / 1，PCP8 | DP8 / TP1 |
| max-model-len | 133120 | 133120 |
| max-num-batched-tokens | 8192 | 240 |
| max-num-seqs | 20 | 60 |
| MTP步数 | 1 | 5 |
| 本地HBM前缀复用 | 关闭 | 关闭 |

基线保留MooncakeConnectorV1进行P→D传输，不接外部池。池化使用MultiConnector组合MooncakeConnectorV1与AscendStoreConnector(memcache)。子connector继承顶层engine_id；不要重新引入互相冲突的独立ID。具体量化、Index Cache、图模式和环境变量直接阅读脚本，不在文档复制另一套完整参数。

## MMC配置与容量

当前归档配置关键值：

```ini
# mmc-meta.conf（69）
ock.mmc.meta_service_url = tcp://192.168.45.69:5000
ock.mmc.meta_service.config_store_url = tcp://192.168.45.69:6000
ock.mmc.meta_service.metrics_url = http://192.168.45.69:8000
ock.mmc.log_level = info

# mmc-local.conf（两机各自文件）
ock.mmc.meta_service_url = tcp://192.168.45.69:5000
ock.mmc.local_service.config_store_url = tcp://192.168.45.69:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 64
ock.mmc.local_service.protocol = device_urma
ock.mmc.local_service.dram.size = 4GB
ock.mmc.local_service.storage.enabled = true
ubsio.disk.path = /dev/nvme3n1p1
ubsio.mem.size_in_gb = 5
ubsio.standalone.device_count = 8
ubsio.standalone.force_new_disk = true
```

Meta读取MMC_META_CONFIG_PATH；模型worker内LocalService读取MMC_LOCAL_CONFIG_PATH，无需额外手动启动LocalService。归档测试注册16个worker、总DRAM为64GiB，容量应以实际注册指标核验，不能把world_size上限当成实际worker数量。UBS的5GB主机缓冲不是池化DRAM容量，也不能当真实SSD读取。

归档配置未显式启用70/60驱逐水位；水位是容量阈值，不是DRAM/SSD命中比例。不要把旧50GB、8GB/worker示例或候选水位当当前配置。每机SSD为本任务专用150GiB分区；force_new_disk=true会初始化缓存内容，必须重新预热。本指南不重新分区、不格式化设备。

## 启动、验证与停止

先检查容器、实际进程归属、在途请求、端口和NPU。只能切换已确认属于本任务且空闲的服务；其他工程占用相同端口时停止本次启动，不抢占。

```bash
W=/home/d00937186/glm53-pooling-lab
# 两机宿主机检查
docker ps --filter name=vllm-pool
npu-smi info
ss -ltnp
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS /dev/nvme3n1p1
```

基线：69执行P、77执行D；健康后69启动proxy。

```bash
bash "$W/scripts/baseline/launch.sh" p       # 69
bash "$W/scripts/baseline/launch.sh" d       # 77
curl -f http://192.168.45.69:8100/health
curl -f http://192.168.45.77:8200/health
bash "$W/scripts/baseline/launch.sh" proxy   # 69
```

池化：69 Meta→两机P/D→健康检查→69 proxy。

```bash
bash "$W/scripts/pcp-pool/launch.sh" meta
# 分别在69、77执行，先核验本机分区归属和初始化安排
CONFIRM_SSD_USE=/dev/nvme3n1p1 bash "$W/scripts/pcp-pool/launch.sh" p
CONFIRM_SSD_USE=/dev/nvme3n1p1 bash "$W/scripts/pcp-pool/launch.sh" d
curl -f http://192.168.45.69:8100/health
curl -f http://192.168.45.77:8200/health
bash "$W/scripts/pcp-pool/launch.sh" proxy
```

以上在宿主机执行，launch.sh通过docker exec与nohup后台运行，不自动创建/启动容器或停止旧服务。按输出的实际路径tail服务日志。原生Meta日志在容器 `/var/log/memcache_hybrid/logs/`，UBS及通信日志按实际运行库配置查找。

```bash
curl -fsS http://192.168.45.69:9000/v1/models
curl -fsS http://192.168.45.69:8000/metrics | grep -E '^memcache_'
curl -fsS http://192.168.45.69:8100/metrics | grep -E 'prefix_cache|external_prefix'
curl -fsS http://192.168.45.69:9000/v1/completions -H 'Content-Type: application/json' -d '{"model":"glm","prompt":"Hello","max_tokens":16,"temperature":0}'
```

短请求成功只证明基础PD链路；长输入、外部池读取、HBM无复用和SSD回温按[测试计划](test-plan.md)分别验证。`/reset_prefix_cache`不等于重置外部DRAM/SSD池。停止顺序为proxy→P/D→Meta，等待在途完成，只对已确认PID操作；保留容器、权重和配置。

## Knowledge Extraction

- [ ] 分介质来源与异常机制仍待进一步核验，不将窗口估算固化为真实命中结论。
