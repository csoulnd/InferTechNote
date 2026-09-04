---
title: "RDMA：远程直接内存访问"
type: concept
domain: foundations
status: active
---

# RDMA：远程直接内存访问

## 核心问题

RDMA 是什么，它为什么能降低服务器之间数据传输的 CPU 开销和延迟？

## 一句话解释

RDMA（Remote Direct Memory Access，远程直接内存访问）是一种让网卡绕过远端 CPU 的数据搬运路径，直接在两台主机已注册的内存区域之间传输数据的网络技术。

## 详细解释

普通 Socket 通信通常经历应用缓冲区、内核协议栈和多次复制；RDMA 由 RNIC 执行 DMA，应用通过 Queue Pair 提交 Work Request，并用 Completion Queue 接收完成结果，从而减少内核参与、内存复制和 CPU 消耗。

RDMA 是能力模型，不等于某一种物理网络。常见承载包括 InfiniBand、RoCE 和 iWARP：InfiniBand 使用专用网络体系，RoCE 在以太网上承载 RDMA，iWARP 基于 TCP。

## 工作原理

1. 应用向 RNIC 注册一段内存，获得访问键。
2. 通信双方建立 Queue Pair，并交换地址、键和连接信息。
3. 应用提交 Send/Receive 或 RDMA Read/Write 请求。
4. RNIC 直接搬运数据，并把结果写入 Completion Queue。

RDMA Read/Write 属于 one-sided 操作：发起端可以访问已授权的远端内存，远端 CPU 不必为每次传输执行匹配的接收逻辑；Send/Receive 仍要求双方协调缓冲区。

## 适用边界

- RDMA 减少数据路径开销，但不消除应用同步、序列化和一致性问题。
- “绕过内核”不等于绕过授权；远端内存必须注册并通过访问键授权。
- RoCE 的稳定低延迟依赖正确的以太网拥塞与无损策略。
- 本文不展开 verbs API、GPUDirect RDMA 或具体交换机调优。

## 实践意义

- 分布式训练、存储和推理 KV Cache 传输都可能受益于 RDMA。
- 评估时同时测量吞吐、尾延迟、CPU 使用率和网络拥塞，不能只看链路带宽。
- 文档中应区分 RDMA 能力、RoCE/InfiniBand 承载和 GPUDirect 等上层组合。

## 应用记录

- 当前作为仓库网络基础术语，等待后续分布式推理或通信设计引用。

## 相关知识

- [通信协议栈](../../agent/concepts/infrastructure/03-communication-protocols.md)

## 参考资料

- [NVIDIA RDMA Aware Networks Programming User Manual](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17)
- [NVIDIA DOCA RDMA](https://docs.nvidia.com/doca/sdk/doca-rdma/index.html)
- [NVIDIA RoCE Documentation](https://docs.nvidia.com/networking/display/rdmacore50/rdma+over+converged+ethernet+%28roce%29)
