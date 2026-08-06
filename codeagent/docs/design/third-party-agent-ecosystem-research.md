# 三方 Agent 生态支持调研

> 调研日期：2026-08-05
> 文档定位：面向 AgentOS 三方 Agent 镜像上架、Windows 本地接入和推理服务供给的第一阶段生态调研。

## 1. 调研背景与目标

AgentOS 接入三方 Agent 生态的目的，不是简单扩充可安装的软件数量，而是以极简方式将不同来源、不同形态的 Agent 纳入统一运行体系。接入后，Agent 可复用一体机统一的身份认证、用户管理、资源分配和推理服务；同时，AgentOS 能结合 Agent、项目和会话信息进行模型路由、实例亲和调度及 KV 前缀复用，实现 Agent 与推理底座的协同优化。

按 Agent Runtime 的控制权，三方 Agent 分为自主部署型、混合运行型和 SaaS 厂商托管型。自主部署型 Agent 可运行在用户控制的服务器、终端或容器中，是 AgentOS 镜像托管和推理协同优化的核心对象；混合运行型 Agent 在本地执行部分操作，但依赖厂商账号、模型或控制面，需根据自定义推理接口能力条件接入；SaaS 厂商托管型 Agent 的 Runtime 完全由厂商控制，AgentOS 主要通过开放 API、连接器、MCP 或 A2A 进行外围互操作。

AgentOS 的三方 Agent 生态支持包含两个相互独立的接入维度：

1. **AgentOS 托管型**：第三方 Agent 能够安装在 Linux 环境中，打包为容器镜像，由 AgentOS 完成镜像上架、实例创建、资源隔离、会话保持、生命周期管理和统一接入。
2. **Windows 本地型**：第三方 Agent 运行在用户 Windows 设备上，AgentOS 不托管其进程，而是向其提供模型推理服务，包括 API Key、Base URL/Endpoint 和模型名称。

场景分类用于组织产品，不能代替上述部署分类。同一个 Coding Agent 可能同时满足 Linux 镜像托管和 Windows 本地运行；桌面 IDE 可能支持 Windows，但不允许配置自定义推理地址，因此不能直接纳入 AgentOS 推理生态。

本文重点回答：

- Agent 能否在 Linux/容器中完整运行并形成可重复构建的镜像；
- 是否支持 TUI、CLI、Headless、Web UI 或桌面 GUI；
- Windows 是否原生支持，还是只能通过 WSL、浏览器或远程桌面使用；
- 是否允许同时配置 API Key、Base URL 和模型名称；
- AgentOS 应通过 SSH/PTY、stdio、HTTP/SSE、ACP、A2A 或其他方式接入；
- 产品究竟是可托管的 Agent Runtime、Windows 本地客户端，还是与厂商生态强耦合的 SaaS。

## 2. 评估口径

### 2.1 AgentOS 接入形态

| 类型 | 定义 | 典型接入方式 |
|---|---|---|
| L1：原生终端托管 | Linux 中运行交互式 CLI/TUI | SSH + PTY、tmux |
| L2：Headless 托管 | 支持单次命令、stdin/stdout 或结构化输出 | 进程调用、JSON/JSONL |
| L3：服务化托管 | Agent 自带稳定服务端，可独立启动 | HTTP、SSE、WebSocket |
| L4：Agent 协议接入 | 具备正式 Agent 互操作协议 | ACP、A2A、Agent SDK |
| L5：完整镜像托管 | 可离线安装、容器化、注入配置并管理生命周期 | AgentOS 镜像与沙箱 |
| W1：Windows 本地供模 | Agent 在 Windows 运行，模型请求发往 AgentOS | OpenAI/Anthropic 兼容 API |

### 2.2 推理接口兼容等级

| 等级 | 判断条件 | AgentOS 适配意义 |
|---|---|---|
| A | API Key、Base URL、模型名均可配置 | 可直接连接 AgentOS 推理网关 |
| B | 可填写 API Key，但 Provider 或 URL 受产品白名单限制 | 需网关模拟预置 Provider 或增加适配层 |
| C | 只能使用厂商账号、订阅或固定云模型 | 无法由 AgentOS 直接提供推理 |
| U | 官方资料未明确 | 必须通过 PoC 验证，不作支持承诺 |

“支持 BYOK”不等于“支持自定义 URL”。MCP 是 Agent 调用工具的协议，ACP 是编辑器调用 Agent 的协议，二者也不能代替模型推理协议。

### 2.3 镜像准入条件

建议只有满足下列条件的产品才进入 AgentOS 镜像上架候选：

- 有明确的 Linux 安装方式或官方容器；
- 核心能力不依赖桌面 GUI；
- 可通过环境变量或配置文件无人工注入模型凭据；
- 项目目录、配置目录和会话目录可以映射到持久卷；
- 进程退出码、日志及健康状态可观测；
- 许可证允许内部部署和镜像再分发，或已取得相应授权；
- 不强制把用户源代码上传到第三方托管沙箱；
- 若只有 TUI，能够在 PTY/tmux 中稳定运行；若支持 Headless，优先使用结构化输出。

## 3. 总体结论

第一阶段最值得优先 PoC 的产品是：

| 优先级 | Agent | Linux 镜像 | Windows 本地 | 推理兼容 | 主要原因 |
|---|---|:---:|:---:|:---:|---|
| P0 | OpenCode | 是 | 是/WSL 更佳 | A | TUI、Headless、Server/Attach、OpenAPI/SSE和自定义 Provider均较完整 |
| P0 | Qwen Code | 是，且有官方镜像 | 是 | A | 国内产品，npm/容器成熟，OpenAI兼容 Base URL 配置明确 |
| P0 | iFlow CLI | 是 | 是 | A | 国内终端 Agent，支持环境变量注入 Key、URL、模型名和 Headless Prompt |
| P0 | CodeBuddy Code CLI | 是 | 是 | A | 国内 Coding Agent，支持自定义 Endpoint、Key、模型及非交互模式 |
| P1 | Claude Code | 是 | 是/WSL、Git Bash | A（Anthropic 协议） | 生态成熟，可经 Anthropic-compatible/LiteLLM 网关接入，但协议不是 OpenAI Chat Completions |
| P1 | Aider | 是 | 是 | A | 安装与容器化简单，模型兼容面广，但没有独立 Agent Server |
| P1 | OpenHands | 是 | 浏览器/容器 | A | 适合云端任务和隔离 Runtime，但整体服务较重 |
| P2 | Gemini CLI | 是，官方容器 | 是 | B/U | Linux/Windows和沙箱成熟，但任意 OpenAI-compatible Endpoint 不是核心官方路径 |
| P2 | Cursor Agent CLI | 是 | 是 | C | Headless 能力较好，但依赖 Cursor API Key，不能直接使用 AgentOS 通用推理 URL |
| P2 | Qoder CLI/IDE | 是 | 是 | B | 可 BYOK，部分 SDK支持 URL override，但产品 UI 主要采用 Provider目录 |
| P3 | Codex | 云端/本地产品面 | 是 | C | Windows体验强，但不是可替换模型后端的通用 BYOK Agent |
| P3 | Trae | 否，桌面 IDE为主 | 是 | C/U | 适合作为 Windows生态观察对象，不适合作为当前镜像上架首选 |

推荐先形成四个镜像基线：`opencode`、`qwen-code`、`iflow-cli`、`codebuddy-cli`。它们与 AgentOS“提供 Key + URL + 模型名”的推理模式匹配度最高。

## 4. Coding Agent 调研

### 4.1 Linux/镜像型重点候选

#### 4.1.1 OpenCode

- **交互**：TUI、非交互 `run`、Web UI、Desktop、IDE、SDK。
- **安装**：npm、独立二进制、Chocolatey、Scoop、Docker；Windows 官方推荐 WSL。
- **镜像适配**：适合。可直接运行 TUI，也可执行 `opencode serve` 或 `opencode web`。
- **分离部署**：官方支持。TUI 可通过 `opencode attach <url>` 连接远程 Server。
- **通信**：HTTP REST、OpenAPI 3.1、JSON、SSE；Server 可配置 Basic Auth。
- **推理兼容**：A。支持 OpenAI-compatible Provider、API Key、自定义 `baseURL`、模型名、请求 Header 和 Body。
- **AgentOS建议**：同时实现两种模式：SSH/PTY 原生 TUI，以及 Server/Attach 服务模式。公网或跨节点通信需在 Basic Auth 外增加 TLS/mTLS 或私网。

官方资料：[安装](https://opencode.ai/en/docs)、[Server协议](https://dev.opencode.ai/docs/server/)、[自定义Provider](https://dev.opencode.ai/docs/providers)。

#### 4.1.2 Qwen Code

- **厂商/区域**：阿里通义，国内优先。
- **交互**：终端交互、单次 Prompt、VS Code 集成、Skills、MCP。
- **安装**：npm、一键安装脚本；官方同时支持 Windows、macOS、Linux。
- **镜像适配**：很好。官方发布 `ghcr.io/qwenlm/qwen-code` 镜像，并支持 Docker/Podman sandbox。
- **推理兼容**：A。`modelProviders` 可定义 OpenAI-compatible、Anthropic、Gemini和本地模型；OpenAI路径可使用 `--openai-api-key`、`--openai-base-url` 或环境变量。
- **AgentOS建议**：作为国内首批基线镜像；配置应使用 `envKey` 引用密钥，避免把 Key 固化进 `settings.json`。

官方资料：[快速开始](https://qwenlm.github.io/qwen-code-docs/en/blog/quickstart/getting-started/)、[模型Provider](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/)、[部署与镜像](https://qwenlm.github.io/qwen-code-docs/en/developers/development/deployment/)。

#### 4.1.3 iFlow CLI

- **厂商/区域**：心流，国内优先。
- **交互**：TUI/CLI、Slash Command、Shell、Sub Agent、MCP、非交互 Prompt。
- **安装**：Node.js 22+、npm或Linux一键脚本；原生支持 Windows，官方建议 Windows Terminal。
- **镜像适配**：适合。Node.js CLI依赖清晰，可通过 npm 离线包和基础镜像构建。
- **推理兼容**：A。官方给出 `IFLOW_API_KEY`、`IFLOW_BASE_URL`、`IFLOW_MODEL_NAME`，并明确支持 OpenAI-compatible API和 CI/CD。
- **AgentOS建议**：列入 P0；PoC重点验证工具调用格式、流式响应、长期会话目录和无浏览器认证路径。

官方资料：[快速开始](https://platform.iflow.cn/cli/quickstart)、[配置与自定义API](https://platform.iflow.cn/cli/configuration/settings)。

#### 4.1.4 CodeBuddy Code CLI

- **厂商/区域**：腾讯云代码助手，国内优先。
- **交互**：交互 CLI、`-p` 非交互、MCP、IDE产品面。
- **安装与系统**：CLI可在 Linux/macOS/Windows使用；Windows存在 Git Bash 检测相关配置。
- **镜像适配**：适合，需固定 CLI版本并验证 npm/原生安装路径。
- **推理兼容**：A。支持 `CODEBUDDY_API_KEY`、`CODEBUDDY_BASE_URL`、`CODEBUDDY_MODEL`，可设置自定义 Header；官方说明可连接 Anthropic-compatible 第三方服务。
- **AgentOS建议**：列入 P0，但推理网关需要提供 Anthropic兼容消息和工具调用接口，不能仅提供 OpenAI Chat Completions。

官方资料：[环境变量和自定义Endpoint](https://www.codebuddy.ai/docs/cli/env-vars)、[CLI参考](https://www.codebuddy.ai/docs/cli/reference)。

#### 4.1.5 Claude Code

- **交互**：交互式 REPL、`-p` 非交互、stdin、IDE集成、MCP。
- **安装**：原生安装方式或 npm；支持 Linux、macOS、Windows 10+，Windows可用 WSL或 Git Bash。
- **镜像适配**：适合，但需评估软件许可、分发方式、自动更新和登录授权。
- **推理兼容**：A，但主要是 Anthropic协议。可通过 `ANTHROPIC_AUTH_TOKEN`、`ANTHROPIC_BASE_URL` 接入 LiteLLM或企业 LLM Gateway。
- **AgentOS建议**：提供 Anthropic-compatible Gateway；保存会话和配置卷；使用 SSH/PTY或 `claude -p` 两条接入路径。

官方资料：[安装](https://docs.anthropic.com/en/docs/claude-code/getting-started)、[CLI](https://docs.anthropic.com/en/docs/claude-code/cli-usage)、[LLM Gateway](https://docs.anthropic.com/en/docs/claude-code/llm-gateway)。

#### 4.1.6 Aider

- **交互**：终端对话、单次 Message、文件监听、实验性浏览器 UI。
- **安装**：uv、pipx、pip、PowerShell脚本、Docker；支持 Windows。
- **镜像适配**：好。官方提供 core/full 两类镜像。
- **推理兼容**：A。可设置 OpenAI API Base，也支持多种云模型和本地 OpenAI-compatible 服务。
- **限制**：没有原生独立 Agent Server；浏览器模式通常仍与执行端同机。
- **AgentOS建议**：优先使用 CLI/Headless，SSH/PTY作为补充；将 Git配置、项目目录和 Aider历史记录持久化。

官方资料：[安装](https://aider.chat/docs/install.html)、[Docker](https://aider.chat/docs/install/docker.html)、[配置项](https://aider.chat/docs/config/options.html)。

#### 4.1.7 OpenHands

- **交互**：Agent Canvas/Web、Cloud、CLI/Headless及集成接口。
- **安装**：当前官方推荐 Agent Canvas，通过 npm或 Docker启动；Cloud免安装。
- **镜像适配**：可行，但比单一 CLI重，涉及 Web服务、Runtime和执行容器。
- **推理兼容**：通常可配置多模型和 OpenAI-compatible 服务，具体字段需在选定版本 PoC中冻结。
- **AgentOS建议**：不应简单嵌入单容器。更适合作为服务型 Agent或独立 Runtime部署，AgentOS管理其外层实例与工作负载。

官方资料：[OpenHands Quick Start](https://docs.openhands.dev/overview/quickstart)。

#### 4.1.8 Gemini CLI

- **交互**：终端、Headless/CI、MCP、IDE关联、容器沙箱。
- **安装**：npm/官方容器；配置路径覆盖 Linux、Windows和macOS。
- **镜像适配**：好，官方提供 `ghcr.io/google-gemini/gemini-cli`。
- **推理兼容**：B/U。官方核心认证路径围绕 Gemini API、Google Cloud和 Vertex AI；不应在未验证前宣称支持任意 OpenAI-compatible Base URL。
- **AgentOS建议**：如果 AgentOS能提供 Gemini/Vertex兼容服务可进一步验证；仅有 OpenAI兼容接口时优先级低于 Qwen Code、OpenCode和 iFlow。

官方资料：[配置](https://github.com/google-gemini/gemini-cli/blob/main/docs/reference/configuration.md)、[容器与Sandbox](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/sandbox.md)。

### 4.2 Windows 本地与桌面型候选

| 产品 | Windows形态 | Key+URL适配 | 是否建议接 AgentOS推理 | 说明 |
|---|---|---:|---:|---|
| OpenCode | 原生CLI、Desktop、WSL | A | 是 | 可直接定义 OpenAI-compatible Provider；Desktop还可连接 WSL Server |
| Qwen Code | PowerShell/终端、VS Code | A | 是 | Windows安装和自定义 Base URL均有官方说明 |
| iFlow CLI | Windows Terminal | A | 是 | 环境变量可直接注入 Key、URL和模型名 |
| CodeBuddy CLI/IDE | CLI及桌面IDE | A | 是 | CLI和IDE均有自定义模型配置，CLI更便于统一管理 |
| Claude Code | WSL或 Git Bash | A/Anthropic | 是，有条件 | AgentOS需提供 Anthropic-compatible Endpoint |
| Aider | PowerShell/终端 | A | 是 | 适合轻量本地客户端，不提供独立 GUI |
| Qoder | Windows IDE和CLI | B | 有条件 | 可填写预置 Provider Key；SDK资料存在 URL override，但需验证最终用户产品是否开放任意 URL |
| Cursor | Windows IDE、Agent CLI | C | 否/需专用适配 | Headless CLI使用 `CURSOR_API_KEY`连接 Cursor云，不是通用模型 Key |
| Codex | Windows桌面、CLI/IDE/云任务 | C | 否 | 产品模型和云任务由 OpenAI服务提供，不属于任意 BYOK Agent |
| Trae | Windows桌面 IDE | C/U | 暂不建议 | 主要是厂商桌面生态；自定义任意 Endpoint能力需以当前版本官方资料为准 |

Qoder支持第三方 Provider API Key，IDE公开列表包括百炼、DeepSeek、智谱、Kimi、MiniMax；这仍属于 Provider白名单式 BYOK。其 CLI SDK资料显示 `CustomModel`可包含 URL，但需要通过实际版本验证 UI和普通 CLI是否同样开放。[Qoder自定义模型](https://docs.qoder.com/user-guide/chat/custom-models)、[Qoder CLI模型](https://docs.qoder.com/en/cli/model)。

Cursor已有 Headless Agent CLI及 JSON输出能力，适合 CI，但认证对象是 Cursor API，不能据此认定它能连接 AgentOS自建推理 Endpoint。[Cursor Headless](https://docs.cursor.com/en/cli/headless)。

### 4.3 推荐镜像公共约定

不同 Coding Agent镜像应尽量暴露统一约定：

```text
/workspace                 用户项目卷
/agent-home                Agent配置、会话和缓存卷
/opt/agent/bin/start       统一启动入口
/opt/agent/bin/health      健康检查
/opt/agent/manifest.json   Agent能力与协议清单
```

建议统一注入以下逻辑变量，再由各镜像启动脚本映射为产品实际变量：

```text
AGENTOS_LLM_API_KEY
AGENTOS_LLM_BASE_URL
AGENTOS_LLM_MODEL
AGENTOS_LLM_PROTOCOL=openai|anthropic|gemini
AGENTOS_WORKSPACE=/workspace
AGENTOS_SESSION_DIR=/agent-home
```

镜像 Manifest至少声明：交互模式、启动命令、Headless命令、PTY需求、健康检查、模型协议、工具调用要求、配置目录、会话目录和许可证来源。

## 5. 通用 Agent 与 Agent 平台

通用终端产品与开发平台必须分开评估。ChatGPT、Kimi、豆包、腾讯元宝、Manus等面向最终用户的 SaaS虽然功能广泛，但通常不能将其 Agent Runtime打包为 AgentOS镜像。AgentOS更应优先接入开源 Runtime或平台 API。

### 5.1 国内优先候选

| 产品 | 产品类型 | 容器/私有部署 | 自定义模型 | AgentOS建议 |
|---|---|:---:|:---:|---|
| Coze Studio开源版 | 低代码 Agent平台 | 是，Docker Compose/Helm | 支持模型管理 | 可作为平台型工作负载，不建议压缩成单Agent镜像 |
| Dify | Agent/RAG/Workflow平台 | 是 | 多Provider及代理能力 | 适合服务型接入和私有部署 |
| FastGPT | 知识库与工作流平台 | 是 | OpenAI-compatible为主 | 适合企业知识和流程 Agent |
| RAGFlow | RAG/Agent平台 | 是 | 多模型 | 适合知识密集型垂直 Agent |
| MaxKB | 企业知识库 Agent平台 | 是 | 多模型 | 国内私有化场景候选 |
| AgentScope | 多Agent开发/运行框架 | 可构建 | 可替换模型 | 更适合成为 AgentOS SDK/Runtime适配对象 |
| Qwen-Agent | Agent开发框架 | 可构建 | Qwen及兼容模型 | 适合基于国内模型构建专用镜像 |
| DB-GPT | 数据与数据库 Agent平台 | 是 | 多模型 | 可归入数据分析垂直生态 |

Coze Studio开源版明确提供 Docker Compose部署、模型管理、Agent/Workflow发布、OpenAPI和 Chat SDK；其服务由前后端及多项基础设施组成，适合按“平台应用”管理，而非每个 Agent复制一套完整平台。[Coze Studio官方仓库](https://github.com/coze-dev/coze-studio)、[API参考](https://github.com/coze-dev/coze-studio/wiki/6.-API-Reference)。

### 5.2 海外框架对照

| 产品 | 形态 | 适合的 AgentOS 接入层 |
|---|---|---|
| AutoGen | Python、多Agent Runtime、Studio Web | SDK/Runtime；分布式场景可考虑 gRPC Worker |
| LangGraph | Python/JS、状态图和服务化 Runtime | SDK/API和长任务状态管理 |
| CrewAI | Python/CLI、多Agent流程 | 将具体 Crew打包为业务 Agent镜像 |
| Flowise | Web/REST、低代码流程 | 服务型 Agent和 API接入 |

AutoGen官方同时提供 AgentChat、Core、Studio和 `GrpcWorkerAgentRuntime`，说明框架型产品与最终用户 Agent的部署粒度不同。[AutoGen官方概览](https://microsoft.github.io/autogen/)。

## 6. 办公 Agent 生态观察

办公 Agent主要依赖宿主软件生态，其核心价值来自组织身份、文档权限、邮件、日历、会议和企业知识，而不是独立 Runtime。因此本节只说明生态边界，不将其作为 AgentOS镜像上架重点。

| 办公 Agent | 强耦合生态 | Windows形态 | 对外开放层 | AgentOS合理接入点 |
|---|---|---|---|---|
| WPS AI | WPS Office、金山文档 | WPS桌面端 | 平台能力/API视版本 | 文档导入导出、业务API |
| 飞书智能伙伴 | 飞书、文档、多维表格、会议 | 飞书桌面端 | Open Platform、Bot、连接器 | IM渠道、文档/表格API |
| 钉钉 AI助理 | 钉钉、宜搭、审批和组织通讯录 | 钉钉桌面端 | Bot、连接器、开放平台 | IM渠道、流程委托 |
| 腾讯文档/会议AI | 腾讯文档、会议、企业微信 | 桌面端/网页 | 腾讯开放能力 | 文档、会议和企业微信渠道 |
| 通义听悟 | 阿里云、钉钉、音视频 | Web/相关客户端 | 云服务API | 转写、总结和会议产物 |
| Microsoft 365 Copilot | Office、Teams、Graph、Entra ID | Windows桌面及Office内嵌 | Graph、Copilot Studio、连接器 | Graph/API、Agent委托 |
| Notion Agent | Notion Workspace和数据库 | Windows桌面/Web | API、Connector、MCP | MCP、页面与数据库API |
| Gemini for Workspace | Gmail、Drive、Docs、Sheets、Meet | 浏览器/Workspace | Workspace API、Google Cloud | Workspace API和企业Agent平台 |

办公 Agent的服务端通常不能由 AgentOS打包或替换。AgentOS应定位为任务来源、执行编排和结果回写方，通过 Bot、API、Connector、MCP或 A2A与办公生态互操作。

## 7. 垂直领域 Agent

垂直领域不宜只按产品知名度筛选，应优先选择能形成可运行镜像的开源 Agent、可调用 API的专业服务，或可在通用框架上复现的行业工作流。

### 7.1 第一阶段候选池（不超过12项）

| 领域 | 候选 | 形态 | 镜像适配判断 |
|---|---|---|---|
| 科研文献 | AMiner | 国内科研知识与分析平台 | SaaS/API生态，通常不是独立Runtime |
| 科研文献 | NotebookLM | Google研究与知识产品 | SaaS，不可镜像化 |
| 科研文献 | Elicit | 文献检索与综述 Agent | SaaS，不可镜像化 |
| 科研文献 | SciSpace | 文献阅读与研究工作台 | SaaS，不可镜像化 |
| 数据分析 | DB-GPT | 开源数据库/数据 Agent | 可容器化，优先 PoC |
| 数据分析 | Chat2DB | 国内开源数据库工具和AI能力 | 可容器化，需确认 Agent自动化深度 |
| 金融投研 | FinGPT | 开源金融大模型/工具生态 | 可构建专用镜像，但不是完整成品 Agent |
| 金融投研 | FinRobot | 开源多Agent金融分析框架 | 可构建，需评估维护活跃度和数据授权 |
| 法律 | MetaLaw/法律开源模型生态 | 模型与数据集为主 | 需在 Agent框架上二次构建 |
| 医疗 | HuatuoGPT/医疗开源模型生态 | 模型与研究项目为主 | 高合规风险，只适合研究验证 |
| 客服营销 | Coze/Dify行业模板 | 工作流和低代码 Agent | 适合以模板+Runtime形成镜像或服务 |
| 软件运维 | OpenHands/Coding Agent运维工作流 | 通用执行 Agent | 可镜像化，需严格权限控制 |

垂直领域当前真正适合 AgentOS的通常不是闭源网站本身，而是“开源框架 + 行业模型 + 行业工具/MCP + 工作流 + 数据授权”组成的可审计镜像。医疗、法律、金融必须额外评估数据许可、结果责任、审计、隐私和监管要求。

## 8. AgentOS 推理服务要求

若希望同时服务 Linux镜像和 Windows本地 Agent，推理网关至少应考虑：

1. **OpenAI-compatible**：`/v1/models`、Chat Completions；根据目标 Agent补充 Responses API。
2. **Anthropic-compatible**：Messages、流式事件、Tool Use、System Prompt和 Token统计，用于 Claude Code、CodeBuddy等。
3. **模型别名**：允许把第三方 Agent期望的模型名映射到 AgentOS实际模型。
4. **工具调用一致性**：不能只兼容纯文本；Coding Agent高度依赖并行工具调用、工具结果、长上下文和流式增量。
5. **自定义Header和租户信息**：支持用户、Agent实例、项目和计费标签透传。
6. **密钥生命周期**：按用户/实例签发短期 Key，支持撤销、配额、审计和最小权限。
7. **网络位置**：Linux容器使用内部服务地址；Windows客户端使用 TLS公网/专网地址，二者不应复用不安全的内网凭据。
8. **能力发现**：向 Agent镜像下发模型上下文长度、输出限制、视觉、推理、Tool Call和协议版本。

## 9. 分阶段实施建议

### 阶段一：四个 P0 Coding Agent

- 构建 OpenCode、Qwen Code、iFlow CLI、CodeBuddy CLI 镜像；
- 对每个 Agent验证 TUI、Headless、Key/URL/模型注入、工具调用、会话持久化和退出恢复；
- 同时在 Windows 进行原生安装验证，确认同一推理网关可复用；
- 固化 Manifest、启动脚本和验收用例。

### 阶段二：协议和体验扩展

- 加入 Claude Code和 Aider；
- 完成 Anthropic-compatible 推理协议；
- OpenCode验证 Server/Attach和 Web UI，比较 SSH/PTY与 HTTP/SSE两种体验；
- 增加统一日志、用量、审批和审计事件。

### 阶段三：平台与垂直 Agent

- 将 Coze Studio、Dify、DB-GPT等按服务型工作负载接入；
- 支持 Agent模板而不是为每个业务复制整套平台；
- 对接 MCP/A2A/业务API；
- 建立行业数据、工具和模型的授权审查流程。

## 10. PoC 验收矩阵

每个拟上架 Agent至少通过以下验证：

| 类别 | 验收项 |
|---|---|
| 构建 | 可从锁定版本和离线依赖重复构建镜像 |
| 启动 | TUI可获得 PTY；Headless可无人工启动 |
| 推理 | 使用 AgentOS Key、URL和模型名成功完成工具调用任务 |
| 文件 | 仅访问授权 Workspace，容器重建后项目不丢失 |
| 会话 | 断开、重连、tmux attach或 Session Resume有效 |
| 安全 | 非root优先、权限可控、密钥不进入镜像层和日志 |
| 网络 | 只允许访问推理网关和批准的工具服务 |
| 可观测 | 有结构化日志、Token/费用、工具调用和错误记录 |
| 升级 | 版本固定、可回滚、配置迁移规则明确 |
| 许可 | 安装、内部使用和镜像分发边界已核实 |

## 11. 风险与待验证事项

- 产品更新较快，安装包、环境变量和登录机制必须按版本冻结；
- “官方提供 Docker sandbox”不一定表示允许将完整产品重新分发为商业镜像；
- OpenAI-compatible实现对 Tool Call、Reasoning和流事件的兼容程度差异很大；
- 某些 Agent的免费登录、搜索、浏览器或固定模型功能无法被自定义 Base URL替代；
- Windows本地 Agent可能同时向厂商遥测、更新或账户服务联网，需要单独披露网络边界；
- TUI字节流透传便于快速接入，但 Gateway无法理解会话语义；长期应优先 Headless结构化输出或公开 Server API；
- 当前 `third-party-agent-agentos-requirements.md` 在部分 Windows终端存在编码显示异常，新文档应统一保存为 UTF-8并在仓库渲染器中验证。

## 12. 结论

AgentOS的三方生态不应简单等同于“支持多少个 Agent名称”，而应形成两条稳定产品线：

1. 将 Linux终端 Agent和服务型 Agent标准化为可审计、可重复构建、可注入模型配置的镜像；
2. 向 Windows本地 Agent提供兼容的推理 Endpoint和短期 API Key，并明确哪些产品真正支持自定义 URL。

首批应优先支持 OpenCode、Qwen Code、iFlow CLI和 CodeBuddy CLI，再扩展 Claude Code、Aider和 OpenHands。办公 Agent保持生态互操作定位；平台和垂直 Agent通过服务、模板、SDK、MCP和 A2A逐步接入。
