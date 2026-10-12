---
title: "Pi Agent 调研摘要"
type: investigation
domain: agent
status: active
---

# Pi Agent 调研摘要

证据基线为 2026-10-10；2026-10-12 调整阅读结构并补充定位与架构分析。未新增部署实验。本次复核固定提交下 P4 的 agent-core 文档，其余基线未变。

## 产品定位：可以由开发者改造和嵌入的 Agent Harness

本报告研究原 pi-mono、现 earendil-works/pi 的 Pi coding-agent 及相关库。Pi 面向希望掌握 Agent 行为的开发者，核心侧重点是提供轻量执行循环、工具、事件和扩展入口，既能本地使用，也能嵌入自己的应用。[P1](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/README.md)[P2](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/packages/coding-agent/README.md)

**研究判断：**它更适合作为“开发者可改造的执行组件”理解。平台团队可以围绕它建设产品，但租户管理、隔离和运营不会因采用 Harness 自动形成。

## 软件架构：模型访问、循环、交互和宿主环境分开

| 关键部分 | 职责与关系 |
|---|---|
| pi-ai | 统一访问模型提供方 |
| pi-agent-core | 处理上下文、工具循环和执行事件 |
| pi-coding-agent / TUI | 提供用户交互、会话和扩展机制 |
| 扩展、技能、SDK / RPC | 调整行为或连接宿主应用 |
| 独立 pi-durable | 实验性的持久化 Harness，应单独评价 |

典型路径是：用户／宿主输入 → 组织上下文 → 转换模型消息 → 模型响应 → 执行工具并产生事件 → 继续循环或结束。宿主决定运行权限和工具环境。[P3](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/packages/coding-agent/docs/extensions.md)[P4](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/packages/agent/README.md)

## 实现基础与设计取舍

已记录基线提供 TypeScript SDK 与库级包关系。消息转换、事件和工具接口是它与宿主连接的重要契约；会话记录、agent-core 内存状态与 pi-durable 任务检查点不是同一机制。[P3](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/packages/coding-agent/docs/extensions.md)[P4](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/packages/agent/README.md)[P5](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/packages/coding-agent/docs/sessions.md)[P6](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/packages/durable/README.md)

**研究判断：**开放扩展与轻量核心提高改造空间，也将更多责任交给宿主：扩展兼容性、进程权限、持久化和调度需要用户设计。工具钩子可以表达策略，但不等于操作系统级隔离。[P1](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/README.md)[P3](https://github.com/earendil-works/pi/blob/42a3497d03ad17e308a2299fa824727894f2c0ec/packages/coding-agent/docs/extensions.md)

## 阅读入口

[详细调研](detail.md)集中说明组件、调用关系、状态、实现依据、部署边界与来源。摘要中的架构取舍属于研究判断；尚未公开或未经实验确认的部分在详细文档中保留。

## Knowledge Extraction

- [x] 完整产品报告采用摘要与详细文档双文档结构，方法已更新到仓库技能。

[返回产品索引](../README.md)
