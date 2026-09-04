---
title: "ACP：Agent Client Protocol"
type: concept
domain: agent
status: active
---

# ACP：Agent Client Protocol

## 核心问题

ACP 是什么，它与 MCP 的职责有何不同？

## 一句话解释

ACP（Agent Client Protocol）是用 JSON-RPC 标准化代码编辑器等客户端与 Coding Agent 之间会话、提示、流式更新、工具调用和终端交互的双向协议。

## 详细解释

ACP 把 Agent 当成可由客户端启动和驱动的独立程序：客户端初始化连接并创建或恢复 Session、发送 Prompt；Agent 通过 Session Update 流式返回文本、思考、工具状态和计划，也可以请求客户端执行文件或终端能力。

ACP 解决“客户端如何操作 Agent”，MCP 解决“Agent 或 LLM 应用如何连接工具和数据”。ACP 的内容块与 MCP 兼容，便于转发工具输出，但两者不能互相替代。

## 工作原理

```text
Editor / Client ← JSON-RPC → Coding Agent
     │                         │
     ├─ 文件与终端能力         └─ 可再连接 MCP Server
     └─ UI、审批与展示
```

## 适用边界

- ACP 不定义模型供应商 API，也不等于 Agent-to-Agent 协议。
- 客户端暴露的文件和终端能力仍需审批与目录边界。
- 协议和能力处于演进中，集成应固定 schema 或 SDK 版本并做兼容测试。

## 实践意义

- IDE 希望替换或接入多个 Coding Agent 时，可用 ACP 统一会话驱动接口。
- Gateway 桥接 ACP 时必须维护 Session、取消、权限与流式事件的双向映射。
- MCP Server 可作为初始化参数注入 Agent，但这不改变 ACP 与 MCP 的职责分工。

## 应用记录

- [OpenCode ACP 集成](../../../../agent/study-notes/opencode/acp-integration.md)
- [ACP 与 MCP 桥接模式](../../integration/acp-mcp-bridge.md)

## 相关知识

- [MCP](mcp.md)
- [SSH Channel 接入](../../integration/ssh-channel.md)

## 参考资料

- [ACP 官方网站](https://agentclientprotocol.com/)
- [ACP 协议文档](https://agentclientprotocol.com/protocol/overview)
- [ACP 官方 Schema](https://github.com/agentclientprotocol/agent-client-protocol/blob/main/schema/v1/schema.json)
