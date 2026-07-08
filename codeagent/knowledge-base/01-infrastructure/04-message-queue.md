# 消息队列 TDMQ / CMQ / DMQ

> 介绍级文档：建立原理认知与场景映射，深入实践见参考链接。

## 原理

**腾讯云 TDMQ**（由原 CMQ/DMQ 演进）提供托管消息队列，核心模型为 **Topic 发布（PUB）** + **消费组订阅**。消息持久化、重试、**死信队列**保证异步链路可靠。

在 Agent OS 中，推理 completion 后产生 **usage 事件**（tenant_id、model、prompt_tokens、completion_tokens 等），同步写 DB 会阻塞 Gateway 响应路径。改为 **PUB 到 Topic**，独立 Consumer 批量落库 MySQL，实现推理与计费解耦。

## 典型应用场景

- vLLM / 网关完成一次推理后，PUB usage JSON 到 `inference-usage` Topic
- Consumer 消费组写入计费表，支持按租户聚合账单
- 消费失败进入死信队列，便于排查与补偿
- 本地开发可用 RabbitMQ / Redis Streams 理解 PUB/SUB 模式后再切 TDMQ

## 参考链接

### 官方标准

- [腾讯云 TDMQ 产品文档](https://cloud.tencent.com/document/product/1409)
- [Topic 发布 / 消费组 / 死信队列](https://cloud.tencent.com/document/product/1409/53564)
- [Python SDK 示例](https://cloud.tencent.com/document/product/1409/56174)

### 视频

- B 站搜索：`TDMQ 消息队列实战`、`腾讯云 CMQ 发布订阅`
