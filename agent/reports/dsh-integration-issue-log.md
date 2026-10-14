---
title: "DSH 穿刺问题清单及进展"
type: work
domain: agent
status: active
---

# DSH 穿刺问题清单及进展

## 1. 文档目的

本记录截至 2026-10-09，作为[DSH 交流议题调研](../investigations/products/deepseek-harness/discussion-agenda.md)的穿刺依据。

本文档仅记录 DeepSeek Harness（DSH）作为三方 Agent 接入 AgentBox 的当前穿刺情况，包括问题现象、责任方、根因、处理方式、验证进展和当前结果。

## 2. 本次穿刺范围

本次穿刺覆盖以下主链路：

```text
DSH 镜像构建
  → AgentBox 镜像上架并配置 web_svc_port
  → 注册中心保存端口并同步 etcd
  → Gateway watch etcd 并监听固定端口
  → 个人智能体读取上架端口并直接跳转
  → DSH 页面和 /api 认证
  → DSH 0.2.x 客户端模块组合资源加载
  → 模型配置
  → /api/remote.mux WebSocket
  → 模型对话和附件上传
  → 设置配置文件入口适配
  → CSV/PDF 等生成文件的 multipart 读取与预览
  → 后续接入文件管理服务下载
```

## 3. 当前总体进展

按独立根因去重后，本次共记录 **16 项有效穿刺问题**。

截至 2026-10-09，固定端口跳转、WebSocket 路由、附件上传、远程设置页适配、DSH 0.2.x 插件资源加载及 multipart 文件预览均已完成穿刺验证。

| 状态 | 数量 | 说明 |
|---|---:|---|
| 已验证完成 | 13 | 修复或兼容方案已在 DSH 镜像、Gateway wheel 或穿刺环境中完成部署验证 |
| 已开发，待部署测试 | 0 | 当前无已开发但尚未部署验证的事项 |
| 已分析，待开发 | 2 | 个人页面配置修改、智能体灰度升级两项能力待开发 |
| 已规划，待服务接入 | 1 | 当前生成文件仅支持 DSH 内预览；下载能力后续统一接入文件管理服务 |

| 主责方 | 问题数 |
|---|---:|
| DSH 镜像 | 5 |
| Manager | 4 |
| Gateway | 6 |
| 文件管理服务（含 DSH 集成） | 1 |

## 4. 问题清单

| # | 问题现象 | 主责方 | 根因 | 处理进展 | 当前结果 | 状态 |
|---:|---|---|---|---|---|---|
| 1 | DSH 启动报 `plugin tree failed to load: @deepseek-ai/dsh-sandbox-local`，前置门显示 `upstream exited (code=1)` | DSH 镜像 | 早期镜像重新组装 DSH npm 依赖时，DSH 主包与传递依赖发生预发布版本漂移，运行时无法解析兼容的 `dsh-sandbox-local`，DSH 启动失败。 | 早期 0.1.5 方案使用 lockfile/override 固定依赖；当前 `dsh-build-1009-1550-proxy` 已升级并精确安装 `@deepseek-ai/dsh@0.2.0-rc.2`，构建末尾同时检查 `dsh-sandbox-local` 的顶层和合法嵌套位置，缺失时立即终止构建。 | 当前 0.2.0-rc.2 源镜像及平台派生后的启动、页面和对话功能均正常 | ✅ 已完成 |
| 2 | 从“个人智能体”点击 DSH 卡片后，仍跳转到 Gateway 主端口下的 `/<agent_name>/` 旧路径，没有进入镜像上架时配置的固定端口 | Manager | 旧的 `build_launch_redirect` 只知道 Gateway 主端口和 `agent_name`，通过拼接 `/<agent_name>/?user_id=...&token=...` 生成地址；它没有从默认版本镜像卡片的 `access_mode[name=web_svc_port]` 中读取上架端口，因此即使上架页已配置 `configPort`，跳转链路也不会使用该值。 | Manager 已改为从默认版本卡片读取唯一的 `web_svc_port`，校验端口后生成 `http://<host>:<web_svc_port>/?user_id=...&token=...`；URL 不再包含 `/<agent_name>/`，也不调用 `3rdagent.web`。同时已处理 `image_process` 的北向端口过滤并补充单元测试。 | 已完成部署验证；个人智能体可按镜像上架配置跳转至对应固定端口 | ✅ 已完成 |
| 3 | DSH 使用未加入信任列表的上架固定端口时，页面可打开，但 `/api/*` 请求返回 403 | DSH 镜像 | DSH connection 模块会校验浏览器请求的 `Host`/`Origin`。上架后浏览器从 `193.65.4.2:19101`–`193.65.4.2:19200` 中的某个固定端口访问；如果该 authority 不在 `trustedHosts` 中，DSH 会在业务 API 执行前拒绝请求。 | 当前 Dockerfile 通过 heredoc 内嵌生成 `/opt/dsh/dsh-host-lan-patch.yml`，保留 `193.65.4.2:19003` 旧入口并覆盖 `19101`–`19200` 端口池，同时支持 `DSH_TRUSTED_HOSTS` 运行时扩展；构建不再依赖外部 YAML 文件。 | 最新镜像已完成动态端口访问验证，Host/Origin 校验正常 | ✅ 已完成 |
| 4 | 通过独立端口打开 DSH 页面后，浏览器内所有 `/api/*` 请求都返回 403 | Gateway | 浏览器请求携带的是外部 authority，例如 `Host: 193.65.4.2:19101`，`Origin` 也基于该地址。旧 Gateway 在反向代理时丢弃了浏览器原始 `Host`，HTTP 客户端随后按沙箱内网 endpoint 自动生成新的 `Host`。DSH 收到的 `Host` 与浏览器 `Origin`、`trustedHosts` 中的外部 authority 不一致，因此触发 Host/Origin 安全校验并返回 403。该问题不是端口未监听，也不是 DSH API 本身报错。 | Gateway 转发请求时显式保留浏览器原始 `Host`，使 DSH 看到的 authority 与浏览器访问地址一致。修改已同步 Gateway 开发并完成穿刺验证。 | 修复已验证成功，当前 Gateway wheel 已包含该逻辑 | ✅ 已完成 |
| 5 | Host/Origin 的 403 修复后，DSH `/api/*` 又返回 401；刷新页面后仍无法建立 DSH 会话 | Gateway | Gateway 和 DSH 各自有独立的认证 Cookie。旧代理为避免向下游泄漏平台 IAM 凭据，直接删除了整个请求 `Cookie`，同时回包转发又没有完整保留上游的多个 `Set-Cookie`。结果是：浏览器用于兑换/维持 DSH 会话的 Cookie 到不了 DSH，DSH 返回的新会话 Cookie 也写不回浏览器。Host 校验通过后，请求真正进入 DSH 认证层，于是错误由 403 变成 401。 | 请求方向只过滤平台 IAM Cookie，保留 DSH 自己的 Cookie；响应方向逐条转发上游 `Set-Cookie`，避免把多个 Cookie 合并或吞掉。修改已同步 Gateway 开发并完成穿刺验证。 | 修复已验证成功，当前 Gateway wheel 已包含该逻辑 | ✅ 已完成 |
| 6 | DSH 设置/模型页面显示 `settings are unavailable in this browser`，无法从远程浏览器读取 Host 模型设置 | DSH 镜像（产品边界） | DSH 上游只把 loopback 页面视为可直接访问 Host 能力。经 Gateway 从非 loopback 地址访问时，`isLoopback=false`，settings 使用浏览器内存模式，因此提示不可用。 | 1008 曾通过 `dsh-client-patch.js` 强制伪装 loopback；当前 `dsh-build-1009-1550-proxy` 已删除该补丁，恢复上游安全边界。模型和凭据改由平台启动配置、环境变量或其他受控方式注入，不再要求远程页面维护 Host 设置。 | 当前提示属于已确认并接受的预期行为；认证、会话、对话和文件功能不受影响 | ✅ 已确认产品边界 |
| 7 | 模型页面报 `provider "火山AI网关" resolves no models`，新建会话或模型调用失败 | Manager（镜像上架配置） | “火山AI网关”是自定义 provider，不在 pi-ai 内置 provider 目录中。只配置 provider 名称或 `baseURL` 时，DSH 无法推导其 API 协议和模型列表，最终解析得到空模型集合。 | 镜像上架启动模板中显式配置 `api: openai-completions`、`baseURL`、非空 `models`，并将 `agent-default-model` 指向同一 provider 和 model。 | 启动模板已验证，`deepseek-v4-flash` 可以被正确解析 | ✅ 已完成 |
| 8 | 浏览器已经发起 `/api/remote.mux` WebSocket，但握手失败；模型不回复，历史加载出现 `reading 'send'` | Gateway | Gateway 处理 WebSocket 时复用了 YuanRong 的 `/serverless/v1/ws` 控制通道 endpoint，再在后面追加 `/api/remote.mux`。该控制通道不是三方 Agent Web 服务的 WebSocket 隧道，所以上游对拼接后的路径返回 404，DSH 的 mux 连接没有建立。 | 已改为基于三方 Agent 的 `/serverless/v1/http` Web endpoint 生成地址，再把 scheme 转换为 `ws/wss` 后转发 `/api/remote.mux`。正式代码已包含 `_agent_web_ws_url()`。 | Gateway wheel 已部署并验证；WebSocket 握手、模型回复和历史加载功能正常 | ✅ 已完成 |
| 9 | 升级 DSH 镜像后，已有用户仍表现为旧版插件行为，新镜像中的修复没有生效；新用户则正常 | DSH 镜像 | AgentBox 会把用户家目录持久化挂载到容器的 `$HOME`，其中包括 `$HOME/.dsh/profiles`。旧入口脚本只在 profile 不存在时播种一次；镜像升级后，已有用户会继续加载持久化目录中的旧插件。 | 最新 Dockerfile 将 profile 骨架保存到 `/opt/dsh/skeleton/.dsh`，并自动使用“`DSH_VERSION` + `plugin/package.json`、`plugin/index.js` 的 12 位源码哈希”生成 revision，格式为 `<DSH_VERSION>-agentbox-<hash>`。入口脚本比较镜像 revision 与 `$HOME/.dsh/.skeleton-rev`；不一致时只同步 `profiles/`，不覆盖 sessions、storages 和凭据。DSH 或插件任一变化都会自动得到新 revision，无需人工递增。 | proxy3 插件已在已有用户目录刷新并验证生效；后续插件更新也能自动触发 profile 刷新，用户数据保持不变 | ✅ 已完成 |
| 10 | DSH 对话中上传附件时，`POST /api/session/uploadFileBinary` 返回 HTTP 502 | Gateway | 浏览器把 `sessionId` 和 `name` 放在请求 URL 的 query 中。旧 `_proxy_http()` 只把 path tail 追加到上游 endpoint，没有把原始 query 合并进去。DSH 收到的上传请求因此缺少定位会话和文件名所需的参数，上游处理失败并由 Gateway 表现为 502。 | Gateway 已新增 `_merge_request_query()` 并接入 `_proxy_http()`，在保留 endpoint 既有 query 的同时转发浏览器业务 query。 | Gateway wheel 已部署并验证；`sessionId`、`name` 等参数可正确转发，附件上传功能正常 | ✅ 已完成 |
| 11 | 旧版远程设置页点击“打开配置文件”后，`openSettingsDocument` 因容器内没有桌面和 `xdg-open` 而失败 | DSH 镜像（产品边界） | DSH 官方“打开配置文件”面向本机桌面环境；AgentBox 运行在无桌面的 Linux 容器中，无法把配置文件交给宿主桌面程序打开。 | 1008 曾修改客户端产物隐藏按钮；当前版本删除 `dsh-client-patch.js` 并恢复上游非 loopback 行为。在当前远程访问场景中，上游页面本身不显示该按钮，因此无需安装 `xdg-utils` 或继续维护客户端补丁。 | 最新镜像中按钮不显示，符合预期；其余页面功能正常 | ✅ 已确认产品边界 |
| 12 | DSH 对话生成文件后，当前仅支持在 DSH 页面中打开和预览，无法下载并保存到用户本机 | 文件管理服务（含 DSH 集成） | 生成文件实际位于服务器工作区。DSH 当前提供的是文件读取和页面预览能力，可能采用分页、大小限制或格式化渲染，不等同于原始文件下载；现阶段尚未接入统一文件管理服务，因此缺少文件登记、权限校验、生命周期管理和浏览器下载链路。 | 本阶段维持“仅支持预览”的现状，不在 DSH 插件内建设临时下载接口。后续接入统一文件管理服务：生成文件登记或归档后取得受控文件标识，由文件管理服务负责用户权限、存储、审计、生命周期及下载传输；DSH 文件卡片再增加下载入口，并通过受控文件标识调用文件管理服务，不向浏览器暴露服务器绝对路径。 | 当前预览功能正常；下载能力已纳入文件管理服务后续建设范围，待服务能力和 DSH 集成方案落地 | 🟣 已规划，待服务接入 |
| 13 | 升级到 DSH `0.2.0-rc.2` 后，首页可以完成认证，但三个 `/plugins/??...` 客户端模块组合地址返回 404，页面功能无法完整加载 | Gateway | DSH 0.2.x 使用 `/plugins/??pkg-a/client.js,pkg-b/client.js&rev=...` 形式加载客户端模块，模块清单位于特殊 query 中。当前 Gateway 向上游转发时丢失该 query，DSH 最终只收到裸 `/plugins`，因此返回 404。问题不是 npm 包缺失，也不是插件未安装。 | 当前镜像通过 `dsh-plugin-agentbox-proxy` 兼容：在 `tapIndex` 阶段把最终渲染 HTML 中的 `plugins/??...` 清单做 base64url 编码并搬到 `__agentbox_plugins_/...` 路径；插件收到路径后解码，再从 loopback 请求 DSH 原始组合资源。Gateway 的长期修改建议是完整保留原始 path 和 query，尤其不要特殊处理或丢弃以 `??` 开头的查询串。 | `dsh-build-1009-1550-proxy` 的最新插件已完成部署验证，模块资源和 DSH 页面功能正常；当前属于 DSH 侧兼容完成，Gateway 根因建议由责任方后续原生修复 | ✅ 已通过 DSH 兼容完成 |
| 14 | CSV 在“表格”模式、PDF 在文档预览中显示“读取失败：client api: workspaceFiles/readBytes failed: Invalid MIME type”；相同文件以代码或纯文本方式可以读取 | Gateway | DSH `/api/workspaceFiles/readBytes` 上游响应为 `multipart/form-data; boundary=...`，正文和状态码均正常。Gateway 构造下游响应时只保留了基础 media type `multipart/form-data`，丢失解析 multipart 必需的 `boundary` 参数。浏览器收到 200 和完整二进制正文仍无法拆分 multipart，因此客户端报 MIME 类型无效。 | Gateway 转发响应时必须优先透传上游完整 `Content-Type` 字符串，不能用解析后的基础 media type 覆盖；建议同步检查 charset 等其他 Content-Type 参数，并增加 CSV、PDF、二进制文件和多 part 响应回归测试。详细证据与建议见 `docs/JiuwenSwarm Gateway multipart响应边界丢失问题分析.md`。 | Gateway 修改已在穿刺服务器验证；响应保留 boundary 后 CSV 表格和 PDF 预览恢复正常。正式代码合入、发布和版本管理由 Gateway 责任方完成 | ✅ 修复验证完成 |
| 15 | 管理面「个人智能体」页面缺少修改配置文件的入口，用户无法自助修改配置文件 | Manager | 个人页面当前只提供卡片查看与启动，未提供配置文件的编辑界面及对应后端接口；配置文件只能由管理员维护或通过启动模板、用户配置目录注入，普通用户无自助修改入口。 | 待开发：新增个人页配置文件编辑入口与保存回写接口，并补充权限校验与审计。 | 当前仅能通过管理员或启动模板维护配置文件，个人页自助修改能力缺失 | 🟡 已分析，待开发 |
| 16 | 智能体升级只能先删除旧版本再重新上架新版本，缺少原地升级与灰度发布能力 | Manager | 当前上架流程没有版本化升级与灰度发布机制，无法在保留旧实例的同时按比例/按实例切换新版本，升级需手工删除后重新上架。 | 待开发：引入版本管理、实例级切换与灰度策略（按用户、按实例或按比例放量），支持平滑升级与回滚。 | 当前升级方式存在服务中断与回滚风险，灰度升级能力缺失 | 🟡 已分析，待开发 |

## 5. DSH 镜像构建方式

GitCode：<https://gitcode.com/xulingfeng/dsh-build>

### 当前构建材料

```text
dsh-build/
├── Dockerfile
└── plugin/
    ├── package.json
    └── index.js
```

| 文件或目录 | 是否必需 | 用途 |
|---|---:|---|
| `Dockerfile` | 是 | 安装 DSH `0.2.0-rc.2` 和系统依赖，安装 AgentBox 前门插件，并内嵌生成远程 patch、profile 骨架和入口脚本 |
| `plugin/package.json` | 是 | 定义 `dsh-plugin-agentbox-proxy` npm 包名、版本和入口，供 `npm pack` 与 `dsh plugin add` 使用 |
| `plugin/index.js` | 是 | 完成 DSH Cookie 交换、Gateway 子路径适配，以及 DSH 0.2.x `/plugins/??...` 模块资源的路径编码代理 |

### 当前服务器构建命令

#### 首次获取代码

```bash
git clone https://gitcode.com/xulingfeng/dsh-build.git
cd dsh-build
```

#### 推荐构建命令

当前 Dockerfile 已内置以下默认值：

- `DSH_VERSION=0.2.0-rc.2`

构建当前版本时只需使用默认的 `DSH_VERSION`。推荐执行：

```bash
docker build \
  --platform linux/arm64 \
  -t dsh:0.2.0-rc.2 \
  .
```

#### 完整命令

升级 DSH、修改插件或排查构建过程时，可以显式传入所有参数：

```bash
docker build \
  --platform linux/arm64 \
  --progress=plain \
  --network=host \
  --build-arg NPM_REGISTRY=https://registry.npmmirror.com \
  --build-arg DSH_VERSION=0.2.0-rc.2 \
  -t dsh:0.2.0-rc.2 \
  .
```

#### 参数含义

| 参数 | 是否必须 | 含义 |
|---|---:|---|
| `docker build` | 是 | 根据当前目录中的 Dockerfile 构建镜像。Dockerfile 使用 heredoc 功能，需要启用 BuildKit；当前新版 Docker 默认已经启用。 |
| `--platform linux/arm64` | 条件必需 | 指定目标镜像为 Linux ARM64。构建服务器本身确定是 ARM64 时可以省略，但显式保留能避免在其他架构机器上误构建。 |
| `--progress=plain` | 否 | 输出完整的逐行构建日志，便于定位 `apt-get`、npm 或插件安装卡住的位置；不影响镜像内容。 |
| `--network=host` | 否，当前环境建议保留 | 让构建步骤使用宿主机网络。此前 Debian 软件源下载曾长时间卡住，保留该参数可复用服务器现有网络和代理路径。 |
| `--build-arg NPM_REGISTRY=...` | 否，当前环境建议保留 | 覆盖 Dockerfile 默认的 npm 官方源。`https://registry.npmmirror.com` 在当前服务器访问更稳定。 |
| `--build-arg DSH_VERSION=0.2.0-rc.2` | 否 | 覆盖要安装的 DSH 版本；当前值与 Dockerfile 默认值相同，所以日常构建可省略。升级版本时必须显式修改或更新 Dockerfile 默认值，并重新完成回归验证。 |
| `-t dsh:0.2.0-rc.2` | 建议必须 | 将构建结果命名为 `dsh`，并设置标签 `0.2.0-rc.2`，供后续 `docker run`、`docker save` 和平台上架使用。该 tag 由构建命令指定，不会根据 Dockerfile 自动生成。 |
| `.` | 是 | Docker build context，表示把当前 `/dsh-build` 目录作为构建上下文，插件目录也从这里复制。

## 6. DSH 启动配置模板

镜像上架时，模型启动配置使用以下 YAML：

```yaml
llm-pi-ai:
  providers:
    火山AI网关:
      apiKeyEnv: MODEL_API_KEY
      api: openai-completions
      baseURL: https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1
      models:
        - id: deepseek-v4-flash
          name: deepseek-v4-flash
          contextWindow: 262144
          maxTokens: 32768

agent-default-model:
  provider: 火山AI网关
  model: deepseek-v4-flash
```

`apiKeyEnv` 填写的是环境变量名。用户 API Key 应通过管理面写入用户配置目录的 `.env`，不应直接写入启动模板。

## Knowledge Extraction

- [ ] 交流议题仍在收敛，后续依据确认的接口与验证结果复核可复用知识，本次保留业务背景和原始证据。
