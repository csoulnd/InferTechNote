---
title: "WorkBuddy 桌面版模型与 Hint 插件调研"
type: work
domain: agent
status: draft
last_updated: 2026-09-04
---

# WorkBuddy 桌面版模型与 Hint 插件调研

## 1. 简介

WorkBuddy 是腾讯推出的全场景 AI 智能工作台，可通过多模型、多专家、Skill 和连接器执行办公、内容、数据及研发任务；产品能力与使用说明见 [WorkBuddy 官方文档](https://www.workbuddy.cn/docs/workbuddy/)。

## 2. 安装部署

WorkBuddy 桌面版采用安装包交付，支持 Windows 10 及以上和 macOS 12 及以上，下载安装后登录即可使用，无需自行部署服务端；参见 [Windows 安装指南](https://www.workbuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Installation-Win-Guide)、[macOS 安装指南](https://www.workbuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Installation-Mac-Guide)和[官方下载页](https://www.workbuddy.cn/app-download/)。

## 3. 使用方式

WorkBuddy 有桌面安装包和 Web/远程入口两种使用方式。本文只关注桌面版；官方明确说明桌面版支持自定义模型，但没有找到纯 Web 端独立配置自定义模型的官方文档，因此暂按“Web 端不承担自定义模型配置”处理。

## 4. 桌面版模型部署

### 4.1 模型来源

| 类型 | 部署或接入方式 | 适用场景 |
|---|---|---|
| 内置模型 | 登录后直接选择，或使用 Auto 自动路由 | 快速使用，无需准备 API Key |
| 套餐模型 | 配置腾讯云 Token Plan/Coding Plan 等提供商 | 已购买模型套餐，希望统一管理额度 |
| 第三方/自建模型 | 配置 OpenAI 兼容 API 或厂商预设 | 使用自有 API Key、企业网关或内网服务 |
| 本地模型 | 本机部署 Ollama，再由 WorkBuddy 连接 HTTP 接口 | 离线、隐私或内网场景 |

官方总览见 [WorkBuddy 模型配置](https://cloud.tencent.com/document/product/1831/134445)。

### 4.2 配置方式

图形界面常见入口为：

    头像 → 设置 → 模型 → 添加模型

也可从对话框底部的模型选择器进入“配置自定义模型”。根据接入类型填写提供商、Base URL/API 端点、API Key、Model ID 和能力标记；非标准接口按需启用“自定义协议”。示例见 [在 WorkBuddy 中配置 Token Plan](https://intl.cloud.tencent.com/zh/document/product/1300/81046)。

### 4.3 配置文件与安全

自定义模型配置保存在本地 models.json。当前资料中存在两个兼容目录：

    ~/.workbuddy/models.json
    ~/.codebuddy/models.json

部署时应以当前客户端实际写入位置和对应版本文档为准。详细字段见 [models.json 配置指南](https://cloud.tencent.com/document/product/1831/134513)。

官方说明 API Key 保存在本地、不上传 WorkBuddy 云端；调用第三方模型时，输入仍会发送至该模型服务。

## 5. Hook 能力

当前 WorkBuddy 的可安装插件系统尚不能确认会从插件包原生加载 Hook。官方页面虽然把 Hook 列为插件类型，但公开的详细 Hook 契约和 hooks/hooks.json 规范主要属于 CodeBuddy；不能据此直接认定 WorkBuddy 插件清单支持 Hook。

不过，WorkBuddy 内部复用的 Agent 运行时会读取本地 Hook 配置。社区已有项目在 WorkBuddy 5.2.6 上验证：可以通过修改 WorkBuddy 用户配置注册 Hook；有的安装器同时安装普通 marketplace 插件，并在插件 Hook 不生效时写入直接 Hook 作为 fallback。因此，hint 接入可以采用“可安装插件 + 本地配置注入”的兼容方案。

与 hint 注入相关的主要事件：

| 事件 | 触发时机 | 用途 |
|---|---|---|
| SessionStart | 会话启动 | 注入一次会话级背景或项目规范 |
| UserPromptSubmit | 用户提交消息后、模型处理前 | 根据当前输入动态检索并注入 hint |
| PreToolUse | 工具执行前 | 根据工具和参数追加约束或阻断调用 |
| PostToolUse | 工具执行后 | 将结果解释或后续建议加入上下文 |
| PreCompact | 上下文压缩前 | 提醒保留关键 hint 或状态 |

对于当前目标，首选 UserPromptSubmit；如有会话级固定背景，再辅以 SessionStart。

Hook 脚本通过 stdin 接收 JSON，通过 stdout 返回 JSON。注入上下文的核心返回值为：

    {
      "continue": true,
      "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": "需要插入模型上下文的 hint"
      }
    }

CodeBuddy 官方文档使用：

    ~/.codebuddy/settings.json
    <workspace>/.codebuddy/settings.json

WorkBuddy 社区实测使用：

    ~/.workbuddy/settings.json

另有较新第三方集成报告使用 ~/.workbuddy-ai/settings.json。安装器不能硬编码单一路径，应先探测现有 WorkBuddy profile，并备份、合并配置。修改后通常需要完全退出并重启 WorkBuddy；是否支持热加载取决于版本。

## 6. Hint 插件设计

### 6.1 目标调用链

    用户提交消息
      → UserPromptSubmit Hook
      → 插件读取 prompt、cwd、session_id
      → 本地规则或 Hint 服务检索
      → 返回 additionalContext
      → WorkBuddy 携带 Hint 调用模型

### 6.2 最小插件结构

    workbuddy-hint-plugin/
    ├── .codebuddy-plugin/
    │   └── plugin.json
    ├── hooks/
    │   └── hooks.json
    └── scripts/
        ├── inject-hint.py
        ├── install.sh
        ├── install.ps1
        └── uninstall.*

plugin.json 使其成为可由本地 marketplace 安装、启用和卸载的插件；hooks.json 保留 CodeBuddy 兼容声明。安装脚本还需把等价 Hook 合并进 WorkBuddy 的 settings.json，使当前不加载插件 Hook 的版本也能生效。

inject-hint.py 从 stdin 读取事件数据，将 prompt 和工作目录交给 Hint 规则或检索器，再输出 additionalContext。诊断日志写入 stderr，stdout 只能输出单个合法 JSON。

该方案不需要进入官方市场，也不需要市场审核。插件可以通过本地 marketplace 目录或自建 Git/HTTP marketplace 分发。安装器需要负责完整生命周期：

1. 探测 WorkBuddy profile 与版本；
2. 安装插件文件并登记 enabledPlugins；
3. 备份 settings.json；
4. 幂等合并自己管理的 Hook；
5. 升级时保留用户其他配置；
6. 卸载时只删除本插件和本插件注册的 Hook；
7. 提示用户重启 WorkBuddy。

不得直接修改 WorkBuddy 的 app.asar 或应用安装目录。这里的“修改本地文件”应限定为用户配置目录；前者容易被升级覆盖，也会破坏签名和安全边界。

### 6.3 设计约束

- **内容边界：** Hint 是额外上下文，不冒充用户原文，需标注来源和用途。
- **失败策略：** Hint 服务失败时默认放行原请求。
- **延迟预算：** Hook 位于模型调用前，应设置短超时和缓存。
- **隐私：** prompt、cwd、transcript_path 可能敏感，非必要不发送到远端。
- **安全：** 外部 Hint 按不可信数据处理，不允许覆盖系统安全规则。
- **上下文预算：** 限制 Hint 条数、字符数或 Token 数。
- **可观测性：** 记录 Hint ID、命中规则、耗时和错误，不记录密钥或完整敏感输入。

### 6.4 验证步骤

1. 在用户级 settings.json 注册一个返回固定 additionalContext 的 UserPromptSubmit Hook；
2. 确认固定 Hint 实际进入模型上下文并影响回答；
3. 将 Hook 封装成可安装插件，并由安装器向 WorkBuddy settings.json 注入直接 Hook；
4. 接入真实 Hint 检索逻辑；
5. 补充缓存、超时、审计和卸载恢复；
6. 验证升级、重复安装和卸载不会破坏用户已有配置。

## 7. 待确认问题

- [ ] 目标 WorkBuddy 版本及其 profile/Hook 配置读取目录；
- [ ] WorkBuddy 当前版本是否忽略插件包内 hooks/hooks.json；
- [ ] 本地 marketplace 插件的安装、启停和卸载流程；
- [ ] UserPromptSubmit 的 additionalContext 是否对所有模型和模式生效；
- [ ] 同一事件多个 Hook 的执行顺序、并行和冲突处理；
- [ ] Hint 最大长度、上下文位置及压缩行为；
- [x] 本地或自建 marketplace 分发不需要进入官方市场审核；
- [ ] WorkBuddy 重启后是否会重写安装器合并的 settings.json。

## 8. 资料索引

- [WorkBuddy 官方文档](https://www.workbuddy.cn/docs/workbuddy/)
- [WorkBuddy 插件系统](https://www.workbuddy.cn/docs/workbuddy/Plugins)
- [Hooks 功能文档](https://cloud.tencent.com/document/product/1831/134517)
- [Hooks 配置参考：CodeBuddy Code 1.16+ Beta](https://cloud.tencent.com/document/product/1831/137030)
- [插件系统](https://cloud.tencent.com/document/product/1831/137027)
- [插件 API 参考](https://cloud.tencent.com/document/product/1831/137036)
- [Guance WorkBuddy OTEL 插件](https://github.com/GuanceCloud/workbuddy-otel-plugin)：可安装插件加直接 Hook fallback 的实证。
- [WorkBuddy Buddy](https://github.com/FlashFamily/workbuddy-buddy)：通过 WorkBuddy 生命周期 Hook 驱动本地应用的案例。
