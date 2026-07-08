# vLLM 推理服务、多租户、API Key 流量统计

> 介绍级文档：建立原理认知与场景映射，深入实践见参考链接。

## 原理

**vLLM** 是高性能 LLM 推理引擎，提供 **OpenAI 兼容 HTTP API**。生产部署时通常以容器或 K8s Pod 运行，通过 `--api-key` 等参数启用多 Key 鉴权。

**多租户**常见两层：vLLM 自身按 Key 区分调用方；前置 **网关**（FastAPI/Nginx/vllm-router）做租户路由、速率限制、usage 聚合。**Prometheus metrics** 可暴露 token 计数，改造后可按 API Key 维度统计流量，支撑计费与配额。

容器内 CodeAgent 通过 HTTP 调用 vLLM，不直接持有模型权重，推理与 Agent 生命周期解耦。

## 典型应用场景

- 容器内 Claude Code / OpenCode 调用统一 vLLM 端点（或云端 Claude API）
- 平台为每个租户分配 API Key，网关记录 prompt/completion tokens
- Prometheus + Grafana 监控各 Key QPS、延迟、token 用量
- usage 事件异步投递 MQ 落库（见 [04-message-queue](04-message-queue.md)）

## 参考链接

### 官方标准

- [vLLM 官方文档](https://docs.vllm.ai/en/latest/)
- [OpenAI Compatible Server + API Key](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Prometheus Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html)

### 教程 / 课程

- [CSDN：vLLM 前置网关区分租户、记录 usage](https://blog.csdn.net/BronzeDragon44/article/details/157009911)

### 开源项目

- [vllm-project/router](https://github.com/vllm-project/router) — 官方路由，多租户 Key + Prometheus
- [baggie11/Multi-tenant-LLM-gateway](https://github.com/baggie11/Multi-tenant-LLM-gateway) — Docker Compose 网关 + PG 计量
- [llm-d/llm-d](https://github.com/llm-d/llm-d) — K8s 生产级多租户（进阶）

### 视频

- B 站搜索：`vLLM 容器化多租户部署`、`vLLM 网关鉴权计费`
