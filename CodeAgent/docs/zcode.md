# 智谱 ZCode 洞察

[![ZCode - 简单、迅捷、氛围十足](../Image/zcode.png)](https://zcode.z.ai/newdocs/welcome)

## 前言

[ZCode](https://zcode.z.ai/newdocs/welcome) 是智谱 AI 推出的面向 **Long Horizon Task（长程任务）** 的全功能 Agentic Development Environment（ADE，智能体开发环境）。它的核心目标，是让 AI Agent 能够端到端、稳定可控地完成跨度更长、步骤更多的开发任务——从需求理解、任务拆解与规划，到编写代码、调试 Bug、项目预览，尽可能在一套工作流中一气呵成。

与早期以「多 CLI Agent 统一调度」为主的定位相比，新版 ZCode 做了更系统的升级：自研 **ZCode Agent** 针对长链路任务的稳定性与上下文保持做了专项优化；以插件形式兼容 Claude / Codex / Gemini / OpenCode 等主流框架；并支持飞书 / 微信 Bot 接入与移动端 Remote 控制，实现随时随地下达指令。平台强调全场景感知（项目结构、文件内容、UI 视觉元素）与分层权限管控，在提升自动化能力的同时保留关键操作的人工确认。

**本文的写作目的**，洞察其 ZCode 设计模式，从产品体验的角度了解其独特之处。

> 官方文档：[欢迎使用全新 ZCode](https://zcode.z.ai/newdocs/welcome)

---

## 第一章 主流 Code Agent 通用五层架构

本章将 OpenCode、Claude Code、Codex 三家官方文档对齐，提炼出一套**最小通用分层模型**。划分原则：从外到内、依赖从低到高；上层只依赖下层接口，不反向耦合实现细节。

> **查证说明**：以下内容依据各产品 2025–2026 年官方文档整理，并标注了与原始草稿的差异修正。
>
> - Claude Code：[How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)、[Glossary](https://code.claude.com/docs/en/glossary)
> - Codex CLI：[Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)、[Core concepts](https://openai-codex.mintlify.app/concepts/overview)、[Architecture Overview](https://www.mintlify.com/openai/codex/architecture/overview)
> - OpenCode：[Server](https://opencode.ai/docs/server/)、[Agents](https://opencode.ai/docs/agents/)、[ACP Support](https://opencode.ai/docs/acp/)

### 1.1 整体架构图

下图展示五层纵向堆叠关系，以及三家产品在**分发形态**上的差异（整包 vs 客户端/服务端分离）。

```mermaid
flowchart TB
    subgraph L1["① CLI / 交互层（Presentation）"]
        direction LR
        CC_CLI["Claude Code<br/>claude 命令 / TUI"]
        CX_CLI["Codex CLI<br/>codex 命令 / TUI"]
        OC_TUI["OpenCode TUI<br/>opencode / attach / acp"]
    end

    subgraph L2["② 会话与协议层（Session & Protocol）"]
        CC_SES["会话管理 · MCP 协议 · 上下文压缩"]
        CX_SES["Thread → Turn → Item<br/>JSONL + SQLite 持久化"]
        OC_SES["HTTP Server · OpenAPI · SSE<br/>ACP（JSON-RPC）· MCP"]
    end

    subgraph L3["③ 编排 / 决策层（Orchestration）"]
        CC_ORC["Agentic Loop<br/>gather → act → verify"]
        CX_ORC["Agent Loop<br/>推理 ↔ 工具调用迭代"]
        OC_ORC["Agent Engine<br/>Build / Plan 双主 Agent · 子 Agent"]
    end

    subgraph L4["④ 工具 / 执行层（Tool & Execution）"]
        CC_TOOL["内置工具 · Shell · 权限门控 · MCP 扩展"]
        CX_TOOL["exec 沙箱 · 文件操作 · MCP · LSP"]
        OC_TOOL["文件/Git/Shell · LSP · 事件总线 · 权限系统"]
    end

    subgraph L5["⑤ 模型适配层（Model Adapter）"]
        CC_MOD["Claude 系列模型 API"]
        CX_MOD["OpenAI Responses API<br/>流式 · Compaction · 重试"]
        OC_MOD["模型无关适配<br/>Claude / OpenAI / Gemini / Ollama …"]
    end

    L1 --> L2 --> L3 --> L4 --> L5

    subgraph PKG_CC["Claude Code 分发：整包（原生安装器 / npm）"]
        CC_CLI --- CC_SES --- CC_ORC --- CC_TOOL --- CC_MOD
    end

    subgraph PKG_CX["Codex CLI 分发：整包（npm 薄包装 + Rust 二进制）"]
        CX_CLI --- CX_SES --- CX_ORC --- CX_TOOL --- CX_MOD
    end

    subgraph PKG_OC["OpenCode 分发：可分离"]
        OC_TUI -.->|HTTP / ACP| OC_SES
        OC_SES --- OC_ORC --- OC_TOOL --- OC_MOD
    end
```

**读图要点**：

- **纵向五层**是三家共有的逻辑架构，不代表物理进程边界。
- **Claude Code / Codex**：五层打包在同一可执行分发单元中，用户只安装一个入口命令（`claude` / `codex`）。
- **OpenCode**：唯一明确支持**客户端与服务端分离**的主流 Agent——TUI、Web、IDE 插件、ACP 客户端均可作为 Layer ①，通过 HTTP 或 JSON-RPC 连接承载 Layer ②–⑤ 的服务端进程。

### 1.2 五层职责详解

#### ① CLI / 交互层（Presentation Layer）

| 维度 | 说明 |
|------|------|
| **职责** | 终端 TUI、参数解析、启动/退出、会话入口、用户输入回显、流式输出渲染 |
| **Claude Code** | `claude` 为统一入口，驱动 REPL / headless（`--print`）及 [Agent SDK](https://code.claude.com/docs/en/how-claude-code-works) 调用；官方术语为 **agentic harness**（智能体外壳），将模型包装为可行动 Agent |
| **Codex CLI** | `codex` 命令由 npm 薄包装（`bin/codex.js`）路由到 `codex-rs` 原生 Rust 二进制；TUI 与 non-interactive 模式共用同一入口 |
| **OpenCode** | `opencode` 启动 TUI 客户端；`opencode attach <url>` 连接远程服务端；`opencode acp` 以子进程方式对接 ACP 兼容编辑器（Zed、JetBrains 等） |
| **可安装性** | Claude Code / Codex：**只能整包安装**，无法单独拆出交互层。OpenCode：TUI 客户端可独立连接已有服务端，但 CLI 包本身仍包含完整能力 |

#### ② 会话与协议层（Session & Protocol Layer）

| 维度 | 说明 |
|------|------|
| **职责** | 会话生命周期、上下文窗口管理、状态持久化、消息编解码、外部协议适配（MCP / ACP） |
| **Claude Code** | 内置会话管理；通过 [MCP（Model Context Protocol）](https://code.claude.com/docs/en/glossary) 连接外部工具与数据源；支持 compaction（上下文压缩）、子 Agent 会话隔离 |
| **Codex CLI** | 采用 **Thread（会话）→ Turn（轮次）→ Item（事件原子单元）** 三级模型；Thread 以 JSONL rollout 文件持久化（`~/.codex/sessions/`），元数据与状态存入 SQLite（`~/.codex/data/`）；支持 `codex resume` 恢复会话 |
| **OpenCode** | 默认 `opencode` 同时启动 **HTTP Server（Hono + OpenAPI 3.1）** 与 TUI 客户端；`opencode serve` 可单独运行无头服务端；支持 SSE 事件流、[ACP 协议](https://agentclientprotocol.com/get-started/introduction)（`session/new`、`session/prompt` 等 JSON-RPC 方法）及 MCP 服务配置 |
| **可安装性** | Claude Code / Codex：会话层**内嵌于整包**，不独立分发。OpenCode：服务端（Layer ②–⑤）可通过 `opencode serve` **独立部署**到本地或远程机器 |

#### ③ 编排 / 决策层（Orchestration Layer）

| 维度 | 说明 |
|------|------|
| **职责** | Agent 主循环、任务拆解、工具选择、多轮路由、子 Agent 调度 |
| **Claude Code** | 官方称 **Agentic Loop**，每轮任务经历 *gather context → take action → verify results* 三阶段，工具结果反馈驱动下一轮决策；扩展点（Skills、Hooks、MCP、Subagents）挂载在循环各阶段 |
| **Codex CLI** | **Agent Loop** 是核心协调层：将用户输入组装为 prompt → 调用 Responses API → 处理流式事件 → 执行工具 → 将结果追加到 input → 循环直至产出 assistant message；支持 `/responses/compact` 自动压缩上下文 |
| **OpenCode** | **Agent Engine** 编排内置 **Build**（全权限开发）与 **Plan**（只读分析）双主 Agent，支持 Tab 切换；另可通过配置定义子 Agent（如 `code-reviewer`、`@general`） |
| **可安装性** | Claude Code / Codex：编排层为内核，**不独立安装**。OpenCode：随服务端一起部署，无单独分发包 |

#### ④ 工具 / 执行层（Tool & Execution Layer）

| 维度 | 说明 |
|------|------|
| **职责** | 文件读写、代码编辑、Git、Shell 执行、沙箱隔离、LSP 集成、权限控制 |
| **Claude Code** | 内置工具集（Read、Edit、Bash、Grep、WebFetch 等）；采用**逐操作权限门控**（permission gating），敏感操作需用户授权；可通过 MCP 扩展工具能力 |
| **Codex CLI** | `codex-exec` 模块负责沙箱执行与文件操作；`codex-rs` 含 `linux-sandbox`、`process-hardening` 等 crate；严格权限模型，支持 workspace-write 等沙箱模式 |
| **OpenCode** | 内置文件 / Git / Shell 工具；集成 LSP 做代码分析；通过权限系统（`permission.edit`、`permission.bash` 等）精细控制各 Agent 能力；事件总线管理诊断信息 |
| **可安装性** | 三家均**不单独分发**工具层；OpenCode 工具层随服务端部署 |

#### ⑤ 模型适配层（Model Adapter Layer）

| 维度 | 说明 |
|------|------|
| **职责** | LLM 调用、提示词模板、流式响应解析、错误重试、多模型路由 |
| **Claude Code** | 适配 Anthropic Claude 系列；统一 Messages API 接口；支持 extended thinking 等模型特性 |
| **Codex CLI** | 通过 `codex-api` crate 调用 **OpenAI Responses API**；处理 SSE 流式事件、Compaction、Memory Summarize；支持 o3 / o4-mini / GPT 系列及第三方后端（Ollama、LM Studio） |
| **OpenCode** | **模型无关**：通过 `opencode auth login` 配置多 Provider，统一适配 Claude / OpenAI / Gemini / Ollama 等；各 Agent 可绑定不同模型 |
| **可安装性** | 均内嵌于分发单元；OpenCode 适配层随服务端部署 |

### 1.3 分发形态对比：「只能安装 CLI」的准确含义

官方文档一致表明：三家产品的**逻辑架构同为五层**，但**物理打包方式**存在本质差异。

```mermaid
flowchart LR
    subgraph A["Claude Code / Codex<br/>垂直集成 · 整包分发"]
        A1["一个安装命令"] --> A2["claude / codex 可执行文件"]
        A2 --> A3["内含 Layer ①–⑤"]
    end

    subgraph B["OpenCode<br/>C/S 架构 · 可分离部署"]
        B1["opencode serve"] --> B2["Server：Layer ②–⑤"]
        B3["opencode TUI"] --> B4["Client：Layer ①"]
        B5["opencode acp"] --> B4
        B6["IDE / Web"] --> B4
        B4 -->|HTTP / ACP| B2
    end
```

#### Claude Code / Codex：闭源垂直集成

| 项目 | Claude Code | Codex CLI |
|------|-------------|-----------|
| **分发形态** | 原生安装器（推荐）或 npm 包，五层合一 | npm 包（`@openai/codex`）内含平台特定 Rust 二进制，亦可直接下载 Release 二进制 |
| **安装命令** | `curl -fsSL https://claude.ai/install.sh \| bash`（推荐）；`npm install -g @anthropic-ai/claude-code`（**已标记 deprecated**） | `npm install -g @openai/codex`；或 `brew install --cask codex` |
| **核心限制** | **无法拆分、无法单独安装某一层**（如只装会话层或工具层） | 同左；`codex-rs` 虽开源可自编译，但官方分发仍以整包为主 |

> **修正**：原始草稿中 Claude Code 包名写为 `@anthropic/claude-code`，官方正确包名为 **`@anthropic-ai/claude-code`**，且 npm 安装已被官方弃用，推荐使用原生安装器。

#### OpenCode：开源 C/S 架构

| 维度 | 说明 |
|------|------|
| **客户端（Layer ①）** | `opencode` TUI、`opencode web` 浏览器端、`opencode acp` 编辑器子进程、`opencode attach` 远程 TUI |
| **服务端（Layer ②–⑤）** | `opencode serve` 启动无头 HTTP 服务（默认端口 4096）；`opencode web` 同时提供 Web UI + 服务端 |
| **安装方式** | 客户端：`npm install -g opencode`（或官方安装脚本）。服务端：随 `opencode serve` 启动，可部署到远程并通过 `OPENCODE_SERVER_PASSWORD` 启用 HTTP Basic Auth |
| **独特之处** | **唯一在官方文档中明确支持「客户端 / 服务端分离部署」的主流 Code Agent**；多客户端（TUI、Web、IDE 插件）可同时连接同一服务端实例，共享会话状态 |

### 1.4 关键结论

| 结论 | 说明 |
|------|------|
| **通用架构 = 五层** | CLI 交互 → 会话协议 → 编排决策 → 工具执行 → 模型适配，是三家对齐后的最小通用模型 |
| **Claude Code / Codex = 整包安装** | 所有内核层打包在单一可执行分发单元中，**没有独立的「内核安装包」** |
| **OpenCode = 可分离安装** | CLI / TUI / ACP 客户端独立存在；内核四层（会话 + 编排 + 工具 + 模型）以 HTTP Server 形式可本地或远程部署 |
| **「只能安装 CLI」** | 专指 Claude Code / Codex 的分发模式——用户通过一条安装命令获得完整五层能力，无法按需拆层采购或部署 |
| **协议层趋同** | 三家均深度集成 **MCP**；OpenCode 额外标准化 **ACP**，使编辑器与 Agent 解耦，这与 ZCode「多框架插件化兼容」的产品方向高度相关 |

### 1.5 与 ZCode 的关联（导读）

理解上述五层模型，是后续分析 ZCode 如何将 Claude Code / Codex / Gemini / OpenCode **以插件形式嵌入同一 ADE 工作台**的基础。ZCode 本质上在 Layer ① 之上构建了统一的交互与 Remote 控制层，并通过插件适配各框架不同的分发与协议特征——这一点将在后续章节展开。

---

> **参考资料**
>
> - [Claude Code — How it works](https://code.claude.com/docs/en/how-claude-code-works)
> - [OpenAI — Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
> - [Codex CLI — Core concepts (Thread / Turn / Item)](https://openai-codex.mintlify.app/concepts/overview)
> - [OpenCode — Server architecture](https://opencode.ai/docs/server/)
> - [OpenCode — Agents (Build / Plan)](https://opencode.ai/docs/agents/)
> - [Agent Client Protocol — Introduction](https://agentclientprotocol.com/get-started/introduction)
