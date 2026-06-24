# TUI/CLI vs ACP 路径体验对照

> opencode / Claude Code 等第三方 Agent 在 SSH+TUI 直连路径与 ACP+WebSocket 转发路径下的体验差异。文档同时覆盖容器化部署方案下的 Gateway ↔ Container 通信层对比。

## 〇、架构背景

两种路径在 Gateway ↔ Container 通信层采用了不同协议：

| 路径 | Client → Gateway | Gateway → Container | Container 内 |
|------|------------------|---------------------|-------------|
| **SSH+TUI** | SSH 加密流 | SSH TCP 字节流透传（ForceCommand → exec ssh） | sshd + tmux + Agent TUI |
| **ACP+WS** | WebSocket / HTTP / IM | ACP JSON-RPC 2.0 over WebSocket | **Adapter（Sidecar）** → Agent 子进程 |

```
SSH+TUI:   用户终端 ──SSH──▶ Gateway :22 ──exec ssh──▶ 容器 :22 ──▶ Agent TUI
ACP+WS:    Web/IM/API ──WS──▶ Gateway ──ACP/WS──▶ Adapter ──stdin/stdout──▶ Agent CLI
```

**核心差异**：SSH 路径 Gateway 只做字节流转发，Agent 原生 TUI 直接服务用户；ACP 路径 Gateway 与容器间走结构化消息，由 Adapter 作为协议翻译层驱动 Agent。

---

## 一、用户输入模式

| 功能 | SSH/TUI 路径 | ACP+WS 路径 | 变化 |
|------|-------------|------------|------|
| **自然语言输入** | 键盘直接输入 | 输入框 → `session/prompt` | 等价 |
| **`@` 取文件** | TUI fuzzy search，文件内容注入 prompt | Web 端文件选择器 → `ContentBlock::Resource` | 等价，体验载体不同 |
| **`!command` 执行 Shell** | TUI 内直接跑，输出追加对话 | ACP `terminal/create` | 等价 |
| **`--flag` 命令行参数** | CLI args（`--model`/`--agent`） | `sessionConfigOptions` 传入 | 等价 |
| **`/editor` 长消息** | 调 `$EDITOR` 写长消息 | Web 文本框天然多行 | Web 更优 |

---

## 二、Slash 命令

| 命令 | SSH/TUI | ACP+WS（opencode 原生 ACP） | ACP+WS（Claude Code 等非 ACP Agent） |
|------|---------|----------------------------|--------------------------------------|
| `/init` | 交互式引导创建 AGENTS.md | 无 ACP 等价物，需 Web 端实现"项目设置" | 同上 |
| `/new` | TUI 内切换 context | `session/new`，Adapter 代理 | Adapter 收到后启新子进程，**等价** |
| `/compact` | opencode 原生压缩上下文 | 无 ACP 标准，需自定义扩展 | **退化，不可用**（除非 Adapter 自行实现） |
| `/undo` / `/redo` | TUI Git 回溯 | 无 ACP 标准 | **退化**，需 Adapter 扩展 |
| `/help` | TUI 显示帮助列表 | Web 端自己渲染 | 等价 |
| `/models` | 切换模型 | `sessionConfigOptions` | 等价 |
| `/sessions` | Agent 自身管理历史 | opencode ACP 原生返回，Adapter 代理 | **退化**，Adapter 需自建 session 注册表 |
| `/thinking` | 切换推理过程显示 | `session/update` + `thought_chunk` | 等价 |
| `/editor` | 调 `$EDITOR` | Web 端不需要 | Web 更优 |
| `/share` / `/unshare` | 分享会话链接 | 无 ACP 标准 | **退化** |
| `/exit` / `/quit` | 退出 TUI | 关闭浏览器/断开 WS | 等价 |
| **自定义命令** | TUI 自动补全 | `available_commands_update` | 等价 |

---

## 三、输出体验（核心差异区）

| 功能 | SSH/TUI | ACP+WS | 变化 |
|------|---------|--------|------|
| **流式输出** | stdout ANSI 字符流 | `agent_message_chunk` 文本块事件 | 等价 |
| **ANSI 颜色 / 语法高亮** | 完整支持（绿色 diff、红色错误） | ❌ 纯文本传输 | **退化**，依赖前端自行渲染 |
| **光标动画 / 进度条** | 旋转器、进度条、内联刷新 | ❌ 丢失 | **退化**，前端可自行实现 loading 动画补偿 |
| **Diff 高亮展示** | 终端原生彩色 diff | 前端可自行实现 diff 渲染组件 | 可补偿 |
| **Tool Call 执行进度** | TUI 内实时文本展示 | `tool_call` 结构化事件 | **ACP 更优**（前端可定制展示） |

> **重要**：ACP `agent_message_chunk` 本身接受任意文本，Agent 的输出不需要是 JSON。Adapter 可以把 Agent stdout 的全部内容原样包裹为 ACP 事件传给 Gateway，前端直接渲染。表现层损失（颜色、动画）是视觉层面的，不影响编码功能，且前端可补偿。

---

## 四、交互与控制

| 功能 | SSH/TUI | ACP+WS | 变化 |
|------|---------|--------|------|
| **权限确认（y/n）** | stdin 输入 y/n | `session/request_permission` 结构化（允许/拒绝/始终允许） | **ACP 更优** |
| **取消操作** | Ctrl+C → SIGINT | `session/cancel` notification | 等价 |
| **自由格式追问**（"用哪个端口？"） | stdin 实时输入任意文本 | ⚠️ `request_permission` 只支持结构化选项 | **退化**（非 opencode ACP 时），Adapter 需模拟 stdin 交互 |
| **键盘快捷键** | TUI 原生 Ctrl+X | Web 端自己绑定，无关 ACP | 等价 |

> **权限确认**是编码场景的高频操作，自由追问极少出现，因此 ACP 在此维度**整体更优**。

---

## 五、会话管理（session lifecycle）

### 5.1 基础会话操作

| 功能 | SSH/TUI | ACP+WS（opencode ACP） | ACP+WS（Claude Code 等） |
|------|---------|----------------------|--------------------------|
| **创建新会话** | TUI 内 `/new` | `session/new`，Adapter 代理 | Adapter 启新子进程，自建注册表 |
| **恢复历史会话** | `/sessions` → 选择 | opencode ACP 原生，Adapter 代理 | Adapter 需自建 session 注册表，记录 session_id → 进程映射 |
| **Fork 会话** | `--fork` 参数 | 无 ACP 标准 | **退化** |

### 5.2 持久化与重连

| 场景 | SSH+TUI | ACP+WS | 变化 |
|------|---------|--------|------|
| **断开重连** | `tmux attach`，无缝恢复 | WebSocket 重连，需 Adapter 恢复 session 上下文 | tmux 更可靠 |
| **容器重启** | tmux 丢失，Agent 进程重建 | Adapter session 也丢失（除非外部持久化） | 都丢失 |
| **跨设备切换** | 必须从终端 SSH | 浏览器/IM/手机随意切换 | **ACP 多端优势明显** |
| **后台执行** | tmux detach，Agent 继续跑 | Adapter 维护子进程，Agent 继续跑 | 等价 |

> **关键结论**：`/new` 在两种路径下无影响。`/sessions` 对于非 ACP Agent（Claude Code 等）需要 Adapter 承担 session 管理职责——这是 ACP+WS 路径最大的额外工程工作。

---

## 六、工具集成

| 功能 | SSH/TUI | ACP+WS | 变化 |
|------|---------|--------|------|
| **MCP 服务器** | TUI 配置 `/mcp` | `session/new` 的 `mcpServers` 参数 | 等价 |
| **Agent 模式切换**（code/ask） | TUI 内选择 | `session/set_mode` | 等价 |
| **Subagent 调用** | opencode 内自动/手动调度 | opencode ACP 下需扩展；Claude Code 无此功能 | **退化**（非 opencode） |

---

## 七、opencode vs Claude Code 在 ACP+WS 路径下的差异

底层 Agent 是否原生支持 ACP Server 模式，决定了 Adapter 的工作量和体验保真度：

| 维度 | opencode（原生 ACP） | Claude Code（仅 CLI） |
|------|----------------------|----------------------|
| **session 管理** | ACP 原生，Adapter 代理 | Adapter 需自建注册表 |
| **流式输出** | `agent_message_chunk` 原生 | stdout 包裹，Adatper 转发 |
| **权限确认** | `request_permission` 原生 | Adapter 解析 stdin 交互 |
| **/compact** | 可扩展实现 | ❌ 不可用 |
| **/undo / /redo** | 可扩展实现 | ❌ 不可用 |
| **Subagent** | 需 ACP 扩展 | ❌ Claude Code 无此功能 |
| **自由追问** | Agent 原生支持 | Adapter 需模拟 stdin |
| **整体保真度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**结论**：opencode + ACP 路径是原生契合，体验接近无损；Claude Code 等纯 CLI Agent 需要 Adapter 承担大量适配工作，session 管理、权限交互等核心链路需要从零建设。

---

## 八、总评

| 维度 | SSH+TUI | ACP+WS + Adapter |
|------|---------|-------------------|
| **终端沉浸体验** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **多端切换** | ⭐ | ⭐⭐⭐⭐⭐ |
| **结构化交互** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Agent 兼容性** | ⭐⭐⭐⭐⭐（零要求） | ⭐⭐⭐（需 Adapter 适配） |
| **工程复杂度** | ⭐⭐⭐⭐（只需 sshd） | ⭐⭐（需开发 Adapter） |
| **前端扩展空间** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 九、覆盖总结

```
全面等价（ACP 原生）:
  session/new, session/list（opencode ACP）, session/load, session/cancel,
  session/set_mode, session/request_permission, file references, streaming,
  tool calls, MCP, terminal/create, thinking display, model config,
  自然语言输入, !command, Slash 命令补全

有差距（需扩展或 Adapter 自建）:
  compact, undo/redo, share, fork, subagent,
  session 管理（非 ACP Agent）, 自由格式追问（非 ACP Agent）

真正丢失（TUI 独有）:
  ANSI 控制符（颜色 / 光标 / 进度动画），tmux 无缝重连
```

---

## 十、核心结论

1. **ACP 覆盖了约 85% 的 TUI 核心能力**。丢失的是终端表现层（ANSI、动画）和少数辅助功能，Agent coding 核心链路没有断裂。

2. **opencode 走 ACP 路径基本无损**。原生 ACP Server 模式使 Adapter 退化为轻量代理。Claude Code 等纯 CLI Agent 需要 Adapter 承担更多适配工作，但 ACP 协议消息链路本身完全兼容任意文本输出。

3. **ACP+WS 方案优于 SSH+CLI** 作为 Gateway→Container 通信协议。SSH 的方案只能做字节流转发或一次性 CLI 调用，无法实现 session 管理、结构化权限确认、流式取消等 ACP 高级能力。ACP+WS 是双向异步事件流，天然适合 Agent 交互场景。

4. **SSH+TUI 和 ACP+WS 并非二选一，而是共存**。SSH 保留了原生终端体验和零适配兼容性，ACP 支撑了 Web/IM/API 多端生态。两者共用同一套容器基础设施，Gateway 根据接入渠道自动选择通信路径。
