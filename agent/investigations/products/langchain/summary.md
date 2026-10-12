---
title: "LangChain 调研摘要"
type: investigation
domain: agent
status: active
---

# LangChain 调研摘要

证据基线为 2026-10-10；2026-10-12 调整阅读结构并补充定位与架构分析。未新增部署实验。本次复核 L1 overview，其余来源沿用原查阅日期。

## 产品定位：给开发者组合 Agent 行为与状态流程的工具

LangChain 面向编写 Agent 应用的开发者，核心侧重点是统一模型和工具接口，并通过 create_agent 与 middleware 组合 Harness。LangGraph 提供较低层状态编排，让确定性业务步骤与模型决策能够放进同一执行结构。[L1](https://docs.langchain.com/oss/python/langchain/overview)[L2](https://docs.langchain.com/oss/python/langgraph/overview)

**研究判断：**理解它应从“怎样用代码表达 Agent 行为和执行状态”出发。相关商业产品 LangSmith、无代码 Fleet 与完整 Harness Deep Agents 需要分别辨认，不能把整个生态混成一个软件包。

## 软件架构：Harness、编排状态与宿主平台分开

| 关键部分 | 职责与关系 |
|---|---|
| LangChain / create_agent | 模型、工具、提示词和 middleware 组合执行循环 |
| LangGraph | 执行状态图、路由、暂停与恢复机制 |
| Checkpointer / Store | 分别保存 thread 状态和跨 thread 数据 |
| Deep Agents | 在相关基础上提供更完整的 Harness |
| LangSmith Deployment / 观测 | 托管执行和分析，部署与遥测位置分别选择 |
| Fleet | 相关无代码 Agent 产品 |

典型代码路径是：应用输入 → Harness／自定义图 → 模型或业务节点 → 工具结果与状态更新 → 下一步或结束。暂停与恢复依赖相应持久化配置；安装框架包不会自动获得云托管能力。[L1](https://docs.langchain.com/oss/python/langchain/overview)[L2](https://docs.langchain.com/oss/python/langgraph/overview)[L3](https://docs.langchain.com/oss/python/langgraph/persistence)[L5](https://docs.langchain.com/langsmith/deployment)

## 实现机制与设计取舍

LangChain Agent 建立在 LangGraph 上；LangGraph 也可独立使用。图状态、跨会话数据和外部业务状态是不同对象，内存 checkpointer 示例不能代表跨进程持久化。[L1](https://docs.langchain.com/oss/python/langchain/overview)[L2](https://docs.langchain.com/oss/python/langgraph/overview)[L3](https://docs.langchain.com/oss/python/langgraph/persistence)

**研究判断：**显式状态和代码扩展提高流程控制力，但用户要理解节点边界、存储和重执行语义。interrupt 恢复时节点从开头执行，外部动作仍需幂等设计。[L4](https://docs.langchain.com/oss/python/langgraph/interrupts) 采用 LangSmith Deployment 后，运营责任再按 Cloud、Hybrid 或自部署模式分配。[L5](https://docs.langchain.com/langsmith/deployment)

## 阅读入口

[详细调研](detail.md)集中说明组件、调用关系、状态、实现依据、部署边界与来源。摘要中的架构取舍属于研究判断；尚未公开或未经实验确认的部分在详细文档中保留。

## Knowledge Extraction

- [x] 完整产品报告采用摘要与详细文档双文档结构，方法已更新到仓库技能。

[返回产品索引](../README.md)
