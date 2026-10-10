---
title: "Pi Agent 产品调研"
type: investigation
domain: agent
status: active
---

# Pi Agent 产品调研

## 范围与结论

此处将用户所称 Pi Agent 解释为原 `badlogic/pi-mono` 的 Pi Harness／coding-agent，未研究 Inflection 的同名聊天产品。旧地址当前跳转到 `earendil-works/pi`。查阅日期：2026-10-10；源码引用为 `42a3497d03ad17e308a2299fa824727894f2c0ec`，未安装或运行。

Pi 是可扩展的 Agent Harness，同时仓库提供模型访问、Agent 循环、TUI 和持久化运行库。它的主要交付形式是本地 CLI 和可嵌入 SDK，而不是与 AgentCore 等价的云托管控制面。[P1][P2]

定制重点是扩展、技能、提示词模板和代码集成。CLI 支持交互、自动化输出与 RPC 控制；SDK 允许嵌入自己的应用。这种扩展性适合平台设计者研究，但不是面向非程序员的可视化低代码平台。[P2][P3]

- [架构及包责任](architecture.md)
- [定制和托管边界](customization-and-hosting.md)
- [来源](sources.md)

## 借鉴点与待验证

研究判断：轻量 Harness、执行事件和宿主环境分离有利于嵌入既有平台。企业采用时仍要另外设计租户身份、隔离与运维。

需实测扩展拦截是否覆盖全部工具入口、RPC 断连后作业行为、会话恢复和版本兼容。当前独立 `pi-durable` 明确标为实验性，需单独验证其重启及工具重放机制，不能把它的保证套到全部 CLI 和 agent-core API。[P6]

## Knowledge Extraction

- [x] 将仓库迁移、包职责及实验性状态纳入调研技能。
- [ ] CLI 与 durable 组合行为待实验。

[返回产品索引](../README.md)
