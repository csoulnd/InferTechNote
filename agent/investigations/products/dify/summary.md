---
title: "Dify 调研摘要"
type: investigation
domain: agent
status: active
---

# Dify 调研摘要

证据基线为 2026-10-10；2026-10-12 调整阅读结构并补充定位与架构分析。未新增部署实验。本次复核固定提交下 D6 的 Agent 技术文档，其余基线未变。

## 产品定位：把模型与工具组织成可维护的业务应用

Dify 面向应用作者、业务流程设计者及平台运营团队，核心侧重点是通过可视化工作流、知识处理、模型和插件，把 AI 能力组合为可发布的应用。新的 Agent 让部分任务可以在流程内部自主完成，也可通过对话构建。[D1](https://github.com/langgenius/dify/blob/0afbe55884d62cb492e4cdbbf470954f252fb474/README.md)[D5](https://dify.ai/blog/why-we-redesigned-dify-agent)

**研究判断：**其架构从“怎样把业务流程做出来”出发。Workflow 控制步骤和输入输出边界，Agent 在任务内部选择动作，两种执行方式共同服务应用交付。

## 软件架构：应用服务、流程执行与 Agent 工作环境分开

| 关键部分 | 职责与关系 |
|---|---|
| Web / API | 编辑、发布和调用应用，管理配置与资源 |
| Workflow / 后台任务 | 组织流程执行及异步工作 |
| Plugin Daemon / 知识处理 | 接入模型、工具及知识能力 |
| Agent Backend | 承载新 Agent 执行 |
| 代码 Sandbox / Agent shell runtime | 两种不同执行环境，应分别分析 |
| 数据库、Redis、文件与向量存储 | 保存不同配置、任务和知识数据 |

典型应用路径是：调用应用 → 按流程执行节点 → 访问模型、工具或知识 → 返回结果；需要新 Agent 时进入其 backend 与 shell 工作环境。部署清单能证明服务存在，不能代替逐条源码调用链。[D3](https://github.com/langgenius/dify/blob/0afbe55884d62cb492e4cdbbf470954f252fb474/docker/docker-compose.yaml)

## 实现基础与设计取舍

在已记录源码基线中，新的 Agent 子系统使用 Agenton 组合 Pydantic AI runs，由 FastAPI 服务承载；Go shellctl 管理运行作业。这些是子系统实现依据，不代表整个 Dify 都基于单一 Agent 框架。[D6](https://github.com/langgenius/dify/blob/0afbe55884d62cb492e4cdbbf470954f252fb474/dify-agent/docs/dify-agent/index.md)[D7](https://github.com/langgenius/dify/blob/0afbe55884d62cb492e4cdbbf470954f252fb474/dify-agent-runtime/README.md)[D8](https://github.com/langgenius/dify/blob/0afbe55884d62cb492e4cdbbf470954f252fb474/dify-agent/docs/agenton/guide/index.md)

**研究判断：**可视化流程和插件降低应用组合成本；多执行路径也使版本、状态、权限与故障处理更复杂。Community、Cloud 和 Enterprise 的部署责任应分别理解，不能把 Enterprise 的隔离描述当作社区默认保证。[D3](https://github.com/langgenius/dify/blob/0afbe55884d62cb492e4cdbbf470954f252fb474/docker/docker-compose.yaml)[D5](https://dify.ai/blog/why-we-redesigned-dify-agent)

## 阅读入口

[详细调研](detail.md)集中说明组件、调用关系、状态、实现依据、部署边界与来源。摘要中的架构取舍属于研究判断；尚未公开或未经实验确认的部分在详细文档中保留。

## Knowledge Extraction

- [x] 完整产品报告采用摘要与详细文档双文档结构，方法已更新到仓库技能。

[返回产品索引](../README.md)
