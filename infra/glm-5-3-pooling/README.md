---
title: "GLM-5.3 双机UB池化部署与测试"
type: moc
domain: infra
status: active
---

# GLM-5.3 双机UB池化部署与测试

- [部署教程](deployment.md)：容器、L2/L3配置及P/D启动。
- [测试计划](test-plan.md)：128K/1K、多前缀90%共享、关闭L1、DRAM/SSD读取80/20的基线设计。
- [返回基础设施索引](../README.md)。

## 脚本与配置

| 资源 | 放置位置与用途 |
| --- | --- |
| [P启动脚本](scripts/start-p.sh) | 69工作区vllm-scripts，MTP步长1 |
| [D启动脚本](scripts/start-d.sh) | 77工作区vllm-scripts，MTP步长5 |
| [统一入口](scripts/start-pool.sh) | 两台vllm-scripts，选择l2/l3及p/d，私有挂载本机UB rootinfo |
| [Meta入口](scripts/start-meta.sh) | 69的vllm-scripts |
| [proxy入口](scripts/start-proxy.sh)与[实现](scripts/load_balance_proxy_server_example.py) | 69的vllm-scripts |
| [L2本地配置](config/l2/mmc-local.conf) / [Meta配置](config/l2/mmc-meta.conf) | 工作区mmc/conf/l2 |
| [L3本地配置](config/l3/mmc-local.conf) / [Meta配置](config/l3/mmc-meta.conf) | 工作区mmc/conf/l3 |

远程工作目录为/home/d00937186，两台不共享。脚本来自本次服务器实际文件；已有容器继续复用，不直接覆盖运行环境。rootinfo/rootinfo-ub.json须使用各机自己的已核验UB拓扑，本仓库不提供跨机通用rootinfo。

## 版本与边界

适用Ascend950DT、CANN9.2.1、vllm-ascend nightly v0.32.0rc A5镜像。P/D脚本保留现场max-num-seqs=20及旧环境变量；教程示例为16并清理了旧变量，尚不能把两个配置混作同一性能结果。测试前按计划统一两组参数。

L3配置指向各机/dev/nvme3n1p1（150 GiB），force_new_disk=true会在启动时初始化该分区。使用前核对设备归属、无挂载、容器访问及配置；不得用于业务数据设备。配置内私网IP、网卡和权重路径均按实际调整。Meta指标端口为8000，未显式设置evict水位；70/60仅为后续测试候选。

已验证：P MTP1/D MTP5、关闭L1后的短请求外部池读写、128K/1K并发1和2、四个独立共享前缀容量标定。69容器已安装独立AISBench环境。真实SSD读取及串行单窗口80/20已取得证据；并发2稳定80/20及正式基线性能对比尚未验收，证据与边界见[测试计划](test-plan.md#501-2026-11-12现场验证进展)。

## 来源

proxy实现沿用vLLM示例，保留其Apache-2.0 SPDX声明。配置来自当前memcache_hybrid安装模板；测试工具参考链接见测试计划。运行日志和权重不提交到仓库。
