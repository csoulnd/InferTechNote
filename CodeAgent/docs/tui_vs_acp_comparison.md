# TUI/CLI vs ACP 路径体验对照

> opencode 核心功能在 SSH/TUI 直连路径与 Web/ACP 转发路径下的体验差异。

## 一、用户输入模式

| TUI 功能 | SSH/TUI 路径 | Web/ACP 路径 |
|---|---|---|
| **自然语言输入** | 键盘直接输入 | 输入框 → `session/prompt`，无区别 |
| **`@` 取文件** | TUI 内 fuzzy search，选择后文件内容自动注入 prompt | Web 端实现文件选择器（树形/搜索），内容塞进 `ContentBlock::Resource`，**体验等价** |
| **`!command` 执行 Shell** | TUI 里直接跑，输出追加到对话 | ACP 靠 `terminal/create` 实现，**效果相同，结构不同** |
| **`--flag` 命令行参数** | 启动时传入 CLI args（`--model`/`--agent` 等） | 初始化时通过 `sessionConfigOptions` 设定，**能力等价** |

## 二、Slash 命令

| 命令 | SSH/TUI | Web/ACP |
|---|---|---|
| `/init` | 交互式引导创建 AGENTS.md | 无 ACP 等价物，需 Web 端实现"项目设置"页面 |
| `/new`（新对话） | TUI 内切换 | `session/new`，**完全等价** |
| `/compact` | 压缩上下文节省 Token | 无 ACP 标准方法，需 `_` 自定义扩展 |
| `/undo` / `/redo` | TUI Git 回溯 | 无 ACP 标准 undo，需 `_` 扩展 |
| `/help` | TUI 显示帮助列表 | Web 端自己渲染帮助页面 |
| `/models` | 切换模型 | `sessionConfigOptions` 实现，**能力等价** |
| `/sessions` | 列出历史会话 | `session/list` + `session/load`，**完全等价** |
| `/thinking` | 切换推理过程显示 | `session/update` 带 `thought_chunk`，**等价** |
| `/editor` | 调 `$EDITOR` 写长消息 | Web 端文本框天然支持多行，**不需要** |
| `/share` / `/unshare` | 分享会话链接 | 无 ACP 标准，需 `_` 扩展 |
| `/exit` / `/quit` | 退出 TUI | 关闭浏览器/断开 WS |
| **自定义命令**（`/test` 等） | TUI 自动补全 | `available_commands_update` 广告命令列表，**体验等价** |

## 三、交互与控制

| 功能 | SSH/TUI | Web/ACP |
|---|---|---|
| **工具执行授权（y/n）** | stdin 原始的 y/n 输入 | `session/request_permission` 结构化授权（允许/拒绝/始终允许），**更优** |
| **流式输出** | stdout ANSI 字符流 | `agent_message_chunk` 文本块事件流，**等价** |
| **光标/进度动画** | 旋转器、进度条（ANSI） | ❌ 丢失，但 Web 端可自行实现 loading 动画补偿 |
| **ANSI 颜色** | 语法高亮、彩色输出 | ❌ 丢失（纯文本） |
| **交互式问答**（Agent 问"用什么端口？"） | stdin 实时输入 | ❌ 不直接支持（`request_permission` 只有结构化选项） |
| **取消当前操作** | Ctrl+C | `session/cancel` notification，**等价** |
| **键盘快捷键**（Ctrl+x 组合键） | TUI 原生 | Web 端自己绑定快捷键，但无关 ACP |

## 四、会话管理

| 功能 | SSH/TUI | Web/ACP |
|---|---|---|
| **创建新会话** | TUI 内 `/new` | `session/new` |
| **恢复历史会话** | TUI 内 `/sessions` 选择 | `session/list` → `session/load` / `session/resume` |
| **Fork 会话** | `--fork` 参数 | 无 ACP 标准 fork，需 `_` 扩展 |

## 五、工具集成

| 功能 | SSH/TUI | Web/ACP |
|---|---|---|
| **MCP 服务器** | TUI 配置 `/mcp` | `session/new` 的 `mcpServers` 参数传入，**等价** |
| **Agent 模式切换**（code/ask/architect） | TUI 界面选择 | `session/set_mode`，**等价** |
| **Subagent 调用** | TUI 内自动/手动调度 | 无 ACP 标准 subagent，需 Proxy Chain 扩展 |

## 覆盖总结

```
全面等价（ACP 原生）:  session/new, session/list, session/load, session/cancel,
  session/set_mode, session/request_permission, file references, streaming,
  tool calls, MCP, terminal/create, thinking display, model config

有差距（需 Web 端自补或 _ 扩展）:  compact, undo/redo, share, fork, subagent

真正丢失（TUI 独有）:  ANSI 控制符（光标/颜色/进度动画）、自由格式交互式问答
```

**核心结论：** ACP 覆盖了 opencode 约 85% 的 TUI 能力。丢失的部分主要是"终端特有的表现层"（ANSI、光标），以及少数未标准化的辅助功能（undo、compact、share）。Agent coding 的核心链路没有断裂。
