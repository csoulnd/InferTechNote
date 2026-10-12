---
title: "Amazon Bedrock AgentCore 调研摘要"
type: investigation
domain: agent
status: active
---

# Amazon Bedrock AgentCore 调研摘要

证据基线为 2026-10-10；2026-10-12 调整阅读结构并补充定位与架构分析。未新增部署实验。本次复核 B2 Harness 与 Runtime 关系，其他来源沿用原查阅日期。

## 产品定位：把 Agent 软件接入可组合的运行基础服务

Amazon Bedrock AgentCore 面向 Agent 开发者与企业平台团队，核心侧重点是提供运行、工具接入、记忆、身份和观测等服务，让用户的 Agent 软件进入受治理的运行环境。它也提供托管 Harness，覆盖希望以配置定义 Agent 的用户。[B1](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)[B2](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html)

**研究判断：**其设计重点在于让执行逻辑与平台原语组合。用户可以自带编排逻辑，也可以使用托管循环；两种路径共享运行平台，但自由度和开发责任不同。

## 软件架构：Harness 在 Runtime 中运行，其他服务按需组合

| 关键部分 | 职责与关系 |
|---|---|
| Runtime | 承载 Agent / 工具的执行环境、会话和运行生命周期 |
| Harness / 用户应用 | 托管 Strands 循环，或自带框架与业务逻辑 |
| Memory / Gateway / 执行工具服务 | 为应用提供上下文、工具接入和执行能力 |
| Identity / Policy | 管理身份与受支持路径的动作权限 |
| Observability / Evaluations 等 | 采集运行行为并支持质量分析 |

典型关系是：调用方 → Runtime Endpoint → Harness 或用户应用 → 模型及按需选择的服务。服务是可组合模块，不能画成每个请求逐层穿过的七层栈。[B1](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)[B2](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html)

## “七层”与实现基础

初始七项服务清单和 AWS 的七维韧性分析模型是不同概念。七维模型描述模型、编排、基础设施、知识库、工具、安全、评估观测等职责；它不是 AgentCore 内部服务部署图，当前模块清单也已超出初始七项。[B5](https://aws.amazon.com/blogs/compute/serverless-icymi-q3-2025/)[B6](https://aws.amazon.com/blogs/architecture/build-resilient-generative-ai-agents/)[B1](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)

Strands 是托管 Harness 的已确认基础，Runtime 保持框架选择空间；示例仓库不是云平台源码。microVM 和 Instances 计算模式应分别解释隔离与持久运行边界。[B2](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html)[B3](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html)[B4](https://github.com/awslabs/agentcore-samples)

**研究判断：**模块组合允许沿用既有 Agent 代码，但用户需要处理服务之间的身份、状态和故障契约；选择 Harness 可减少循环集成工作，同时接受其编排边界。

## 阅读入口

[详细调研](detail.md)集中说明组件、调用关系、状态、实现依据、部署边界与来源。摘要中的架构取舍属于研究判断；尚未公开或未经实验确认的部分在详细文档中保留。

## Knowledge Extraction

- [x] 完整产品报告采用摘要与详细文档双文档结构，方法已更新到仓库技能。

[返回产品索引](../README.md)
