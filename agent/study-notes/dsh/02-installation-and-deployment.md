---
title: "DeepSeek Harness 安装与部署"
type: work
domain: agent
status: draft
---

# DeepSeek Harness 安装与部署

## 1. 先选运行方式

| 场景 | 推荐方式 | 命令入口 |
|---|---|---|
| 快速体验 | npm 临时执行 | `npx @deepseek-ai/dsh web` |
| 学习与贡献源码 | 克隆仓库、pnpm 构建 | `pnpm dsh web` |
| CI/一次性任务 | Headless profile | `dsh --profile headless "任务"` |
| 程序集成 | TypeScript/Python SDK profile | `sdk` / `sdk-minimal` |
| 编辑器/自动化协议 | ACP profile | `acp` |

官方开发基线要求 Node.js 22.19+ 或 24+，仓库在基线提交中固定 `pnpm@11.7.0`；Git 要求 2.26+。快速运行只需满足发布包的 Node 要求，源码开发还需 Corepack/pnpm 与 Git。

## 2. npm 快速启动

```bash
npx @deepseek-ai/dsh web
```

默认监听 `http://127.0.0.1:3080` 并尝试打开浏览器。常用参数：

```bash
npx @deepseek-ai/dsh web --no-open
npx @deepseek-ai/dsh --profile web --port 8080
```

Web 模式不接受 `--host 0.0.0.0`。远程使用应保留 loopback 监听并采用 SSH 端口转发，例如：

```bash
ssh -L 3080:127.0.0.1:3080 user@server
```

服务端运行 `npx @deepseek-ai/dsh web --no-open`，本地访问 `http://127.0.0.1:3080`。

## 3. 从源码运行

```bash
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
corepack enable
pnpm install
pnpm run typecheck
pnpm run build
pnpm dsh web
```

`pnpm run build` 生成包与前端产物；`pnpm dsh web` 使用已经构建的产物，不会自动重建。修改源码后的日常检查优先运行相关包测试，再运行仓库级 `typecheck`。

为了与本资料完全一致，可在学习分支固定基线：

```bash
git switch --detach 76fda729799fe9b3848dbe2c211d4b231032b81e
```

这是只读学习最可复现的方式；准备贡献时再切回官方最新分支。

## 4. 第一次配置模型

Web UI 中进入“设置 → 模型”，配置 DeepSeek 或其他 Provider。DeepSeek 密钥可由 `DEEPSEEK_API_KEY` 提供，也可在 UI 中保存。UI 保存的凭据位于 `$DSH_HOME/.credentials.yaml`，前端只读取脱敏描述，不回传明文。

凭据解析来源包括继承环境、Harness 凭据文件、启动目录 `.env` 与 `$DSH_HOME/.env`。不要把真实密钥提交到项目仓库。

自定义 OpenAI-compatible 网关需要配置 provider id、base URL、协议、凭据与模型。若网关不接受 OpenAI 默认请求形状，可在 `$DSH_HOME/settings.yaml` 中调整兼容项，例如：

```yaml
llm-pi-ai:
  providers:
    company-gateway:
      apiKeyEnv: COMPANY_LLM_API_KEY
      api: openai-completions
      baseURL: https://gateway.example/v1
      compat:
        supportsDeveloperRole: false
        maxTokensField: max_tokens
      models:
        - id: my-model
```

## 5. Profile、配置与排障

默认 Harness home 是 `~/.dsh`，可通过 `DSH_HOME` 改变。常见文件：

```text
$DSH_HOME/
├─ settings.yaml
├─ .credentials.yaml
├─ cordis.patch.yml
└─ profiles/
   └─ web/
      ├─ package.json
      ├─ cordis.yml
      └─ cordis.patch.yml
```

先看最终配置，再猜问题：

```bash
npx @deepseek-ai/dsh --profile web --dump-default-config
npx @deepseek-ai/dsh --profile web --dump-config
```

前者只包含 bundle 层；后者还包含 profile、home 和 `--patch` 覆盖，并在注释中标出来源。patch 按 id 替换整行 `config`，修改一个字段时需要保留该行仍需的其他字段。

安装外部插件或 bundle：

```bash
dsh plugin --profile web add <package-or-git-spec>
dsh plugin --profile web remove <package>
```

CLI 会在对应 profile 目录调用 pnpm，并根据依赖包 `package.json` 的 `dsh.bundle` 声明维护 bundle 层。第三方插件是可执行代码，应先审查源码、依赖和权限。

## 6. Headless 与自动化

```bash
DEEPSEEK_API_KEY='...' npx @deepseek-ai/dsh --profile headless "总结当前仓库"
```

Headless 创建一个新的持久会话，等待 Agent 停稳并 flush；推理增量写入 stderr，最后文本写入 stdout。最终 `turn/end` 为 `completed` 时退出码为 0，否则为 1，适合 shell/CI 判断结果。它不启动 HTTP 服务或浏览器。

## 7. 部署安全基线

- 在一次性 VM、容器或专用主机运行高风险任务；内置沙箱和审批不能替代强隔离。
- 工作区、文件、网络、进程和密钥均按最小权限开放。
- 不把 Web 端口直接暴露到公网；远程优先用 SSH 转发或受控反向代理。
- 锁定 DSH 与第三方插件版本，升级前用 `--dump-config` 比较组合变化。
- 备份可写工作区和 `$DSH_HOME`；对敏感环境禁用不需要的工具与遥测。
- 配置启停、失败、权限拒绝和工具调用都应进入外部审计/日志系统。

## 8. 常见故障

| 现象 | 检查 |
|---|---|
| `MISSING_CREDENTIAL` | 环境变量名、`.credentials.yaml` 引用、启动目录 `.env` |
| `UNKNOWN_MODEL` | Provider 是否保存、model id 是否存在、会话是否仍绑定旧路由 |
| 网关 401 | 密钥与 base URL；模型发现端点可能不受支持，可手工录入 |
| 网关拒绝所有请求 | `supportsDeveloperRole`、`maxTokensField` 等 compat 配置 |
| 插件不启动 | `--dump-config` 中是否存在、`inject` 依赖是否有 Provider、schema 是否通过 |
| 修改源码不生效 | 是否重新 `pnpm run build`；Web 客户端 watcher 是否启动 |
| patch 修改丢字段 | patch 是整行 config 替换，不是深度合并 |

## 9. 参考

- [官方中文 README](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/README.zh.md)
- [开发指南](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/development.zh.md)
- [CLI 行为参考](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/apps/cli/reference/README.zh.md)
- [模型配置指南](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/user/guide/providers.zh.md)
- [Web UI 指南](https://github.com/deepseek-ai/deepseek-harness/blob/76fda729799fe9b3848dbe2c211d4b231032b81e/docs/user/guide/index.zh.md)
