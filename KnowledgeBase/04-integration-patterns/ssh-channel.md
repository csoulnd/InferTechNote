# SSH Channel 接入

## 两种 SSH 用法

### 1. TUI 字节流透传（不经 MessageHandler）

用户 SSH 连接 Gateway → Gateway 将终端 I/O 原样转发到 Agent 容器 sshd → 容器内运行原生 `claude` / `opencode` TUI。

**特点**：
- Gateway **不解析**终端内容
- 支持交互式输入、流式输出、tmux 会话
- 体验等同直连服务器

### 2. 消息路径 CLI exec（经 MessageHandler）

Web / IM Channel → MessageHandler 双队列 → asyncssh 在容器内执行：

```bash
opencode run --no-tui "$用户消息"
# 或
claude --print "$用户消息"
```

响应经 `_robot_messages` 回投 Channel。

## 标准

- [RFC 4254 SSH Connection Protocol](https://datatracker.ietf.org/doc/html/rfc4254)

## 选型

| Agent | TUI 透传 | CLI exec |
|-------|----------|----------|
| Claude Code | ✅ | ✅ `claude --print` |
| OpenCode | ✅ | ✅ `opencode run --no-tui` |

Claude Code 无 HTTP Server，SSH + CLI 是通用方案。

## agentos

- `jiuwenswarm/gateway/` — SSH channel 实现
- 对照 [00 全局架构](../00-architecture-overview.md) 路径 A/B 时序图

走读笔记：[notes/integration-ssh-passthrough.md](../notes/integration-ssh-passthrough.md)
