---
title: "LangChain 定制及企业托管"
type: investigation
domain: agent
status: active
---

# LangChain 定制及企业托管

## 定制能力

`create_agent` 面向代码作者，通过 middleware 改变模型调用、工具策略和上下文；LangGraph 提供更细的节点、路由与状态设计。[L1][L2] 接口简短不等于可视化低代码。Fleet 才是生态中明确面向无代码构建的产品，原 Agent Builder 已更名为 Fleet。[L7]

因此调研“LangChain 是否支持低代码”时，需要分别回答框架代码入口与 Fleet 产品入口；还需确认可迁移配置、支持工具与套餐，不能推定两者具有完全同构的定制能力。

## 企业托管责任

官方 LangSmith Deployment 支持以下部署环境：[L5]

| 环境 | 控制面 | 数据面及运维 |
|---|---|---|
| Cloud | 服务商 | 服务商承载 Agent Server 与数据库 |
| Hybrid | 服务商 | 用户基础设施承载数据面 |
| Self-hosted with control plane | 用户 | 用户 Kubernetes；需要对应 Enterprise 授权 |
| Standalone server | 无部署控制面 | 用户运营 Server、PostgreSQL、Redis 等并核对许可 |

观测平台放在哪里与执行部署放在哪里是两个问题。选择私有数据面时仍需检查 traces、模型请求和外部工具的数据路径。

纯 LangChain／LangGraph 库提供程序机制；宿主仍负责用户身份、网络、隔离、容量、凭据和备份。LangSmith Deployment 承担的运营责任则随模式变化，不能依据“框架开源”推定整个商业托管栈都可无条件自部署。

## 恢复和人工审批

LangGraph interrupt 配合 checkpointer 与同一 thread 标识，可暂停等待输入并恢复。恢复时节点会从开头重新运行，interrupt 前的代码也会再次执行。[L4]

例如把“写业务数据库”放在 interrupt 前，恢复可能再次写入。研究判断：应设计持久化操作结果、幂等键或独立动作节点，并验证重启窗口。这不是 LangGraph 自动替外部系统实现的事务保证。人工审批还需要宿主验证审批者身份和动作内容，不能仅信任传回的布尔值。

## Knowledge Extraction

- [x] 分开库的执行语义和平台运营责任，纳入技能。
- [ ] 恢复、审批身份与外部幂等待实验。

[来源](sources.md) · [产品概览](README.md)
