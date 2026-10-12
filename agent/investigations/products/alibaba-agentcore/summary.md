---
title: "阿里云 AgentCore 调研摘要"
type: investigation
domain: agent
status: active
---

# 阿里云 AgentCore 调研摘要

证据基线为 2026-10-10；2026-10-12 调整阅读结构并补充定位与架构分析。未新增部署实验。本次复核 A1 产品概述，其他来源沿用原查阅日期。

## 产品定位：统一组织和治理多来源 Agent

阿里云 AgentCore 面向需要创建、接入和统一管理智能体的开发者与企业平台团队。其核心侧重点是把 Agent、模型、技能、工具、凭证与 Team 放进统一资源体系，再连接运行环境和业务渠道。[A1](https://help.aliyun.com/zh/agentcore/agentcore-product-overview)[A2](https://help.aliyun.com/zh/agentcore/agentcore-basic-concepts)

**研究判断：**理解它首先应看“企业如何组织和管理 Agent”，再看单个 Agent 怎样运行。Workspace 和 Team 是这一定位的重要对象；它既有配置式 Harness，也保留高代码和已有 Agent 纳管路径。

## 软件架构：资产控制与会话执行分开

| 关键部分 | 职责与关系 |
|---|---|
| Workspace / Agent / Team | 定义资产归属、执行单元与协作组织 |
| QwenPaw / Qoder 或自定义应用 | 承担模型决策和工具循环；不同模式能力不同 |
| 实例 / Session / Endpoint | 承载运行、组织连续交互并对外提供入口 |
| 模型、Skill、MCP、凭证 | 为执行逻辑提供能力与外部资源 |
| 观测及网络等云依赖 | 支撑治理和运行接入 |

典型请求的研究视图是：业务渠道 → Endpoint → 实例中的 Session → Harness 或用户应用 → 模型与工具 → 返回结果。Workspace 管理配置，不是每次请求必须经过的处理层；Team 也不是所有单 Agent 调用的必经环节。[A2](https://help.aliyun.com/zh/agentcore/agentcore-basic-concepts)[A3](https://help.aliyun.com/zh/agentcore/agent-instructions-managed-harness)[A4](https://help.aliyun.com/zh/agentcore/high-code-agent-deployment-guide)

## 开发基础与设计取舍

公开文档确认托管 Harness 引擎与 SDK 接口；完整平台的内部服务拓扑、数据库选型及底层沙箱实现仍未确认。CMS、PrivateLink、SchedulerX3 是公开列出的依赖，不能据此推定整个平台基于某一个框架或 Kubernetes。[A1](https://help.aliyun.com/zh/agentcore/agentcore-product-overview)[A3](https://help.aliyun.com/zh/agentcore/agent-instructions-managed-harness)[A5](https://help.aliyun.com/zh/agentcore/agentcore-sdk-overview)

**研究判断：**统一资产和多种接入模式有利于管理既有 Agent，但统一管理界面不代表执行语义一致。要理解这套设计，应沿着“资产绑定 → 执行引擎 → 会话状态”的关系阅读，而不是把功能分组当作垂直七层。

## 阅读入口

[详细调研](detail.md)集中说明组件、调用关系、状态、实现依据、部署边界与来源。摘要中的架构取舍属于研究判断；尚未公开或未经实验确认的部分在详细文档中保留。

## Knowledge Extraction

- [x] 完整产品报告采用摘要与详细文档双文档结构，方法已更新到仓库技能。

[返回产品索引](../README.md)
