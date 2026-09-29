---
title: "第三方智能体上架与启动跳转流程"
type: design
domain: agent
status: draft
date: 2026-09-29
---

# 第三方智能体上架与启动跳转流程

## 1. 范围与依据

**当前实现**：

| 图组 | 链路 | 起点 | 终点 |
|---|---|---|---|
| 图一 | **上架** | 管理员拿到制品（npm tgz / OCI 镜像归档） | 注册中心上出现一张可查询、可启动的卡片 |
| 图二 | **启动跳转** | 用户在管理面点击启动 | 浏览器进入实例内的 Web 服务（`access_mode.web`） |


关联文档：

- 上架的字段模型与镜像工厂设计见 [image-factory v2](third-party-agent-artifact-image-factory-design-v2.md) §5.1 / §6.1 / §8.3 / §8.4；本文补的是**实际调用链、制品状态机，以及两条链路的耦合点与变动点分析**。
- 跳转的协议缺口与理想模型见 [接入架构](third-party-agent-access-architecture.md) §2.3 / §2.4；本文画的是**实测时序**，含隐藏 iframe 预热与 `launch-status` 轮询这两个前端行为。

## 2. 组件与关键Tag

| 组件 | 地址 | 职责 |
|---|---|---|
| 管理面 Manager | `:8090` | 制品的通用门禁与台账、launch 配置渲染、卡片查询转发、返回 303 |
| 镜像工厂 image-process | `:8091` | 解析制品、选 Recipe 与 Base、构建镜像、产出 `runtime_spec` |
| 注册中心 a2x-registry | `:4003` | 卡片（镜像记录）与实例（Runtime）的权威 |
| Gateway | `:19000` | Web/WS 接入、鉴权、实例解析与代理；南向调 YuanRong |
| Gateway SSH Channel | `:2222` | SSH 北向接入，中继到实例 PTY |
| YuanRong Frontend | `:8888` | 沙箱生命周期（`/api/agent`）与实例 HTTP/WS 代理（`/serverless/v1/...`） |

| 标识 | 生成方 | 含义 |
|---|---|---|
| `name` / `framework` | 管理员填写（注册中心上两者同值） | 卡片名；同时是 Gateway 侧 `agent_type` 与 URL 路径段 `/{agent}/` |
| `version` | 镜像工厂解析 | 同一 name 的版本；用户视图按**默认版本**启动 |
| `access_mode[]` | 管理员填写 | `{name, port, cmd}`：`web` 决定浏览器跳转的端口与启动命令，`tui` 决定是否注入 sshd 与 SSH 接入 |
| `launch 模板` | 管理员填写，随制品落 sidecar | 启动时渲染进用户 home 的配置文件；`${MODEL_API_KEY}` 是 agent 侧环境变量引用，由用户单独配置 |
| `service_id` | Gateway 派生 `generic_{sha256(user\0framework)[:8]}` | 注册中心实例行主键：一个 (用户, 框架) 恒定一行，POST 幂等 upsert |
| `instance_id` | YuanRong `POST /api/agent` | 沙箱实例 ID；代理、SSH、删除都用它 |
| `content_digest` | 管理面（SHA-256） | 制品身份；launch sidecar 与本地台账按它绑定 |

## 3. 图一：第三方智能体上架

### 3.1 上架流程图

**简略流程**

```mermaid
flowchart LR
    A["管理员上传制品<br/>tgz / OCI 归档"] --> B["管理面通用门禁<br/>落盘 packages/{digest}.artifact"]
    B --> C["镜像工厂构建<br/>image_ref + runtime_spec"]
    C --> D["注册中心落卡<br/>POST /api/images"]
    D --> E["卡片可查询、可启动"]
    E -.->|"无实例时"| F["删卡"]
```

**完整流程**

```mermaid
flowchart TD
    A["管理员：选择制品并填写 name / description / access_mode<br/>可选填 launch 模板"] --> B{"上传方式"}
    B -->|"一次上传"| B1["POST /api/v1/thirdparty_agent/cards<br/>multipart：package + access_mode + name + description"]
    B -->|"分片上传"| B2["POST /uploads<br/>PUT /uploads/{id}/parts/{n}<br/>POST /uploads/{id}/complete<br/>POST /cards/from-artifact"]
    B1 --> C
    B2 --> C
    C["管理面通用门禁：大小 / 安全命名 / 落盘<br/>packages/{digest}.artifact<br/>台账状态 = uploaded，建 BuildTask（并发与重复构建锁）"] --> D["绑定 launch 配置 sidecar（以 digest 为键）<br/>返回 202 + request_id"]
    D --> E["异步任务 _run_build：状态 = building"]
    E -->|"制品路径"| F

    subgraph FACTORY["镜像工厂 image-process:8091（Recipe 可扩展）"]
        direction TB
        F["RecipeRegistry.resolve：按制品类型选 Recipe"] --> F1["解析制品：name / version / os / arch"]
        F1 --> F2["平台校验：image 的 os/arch 必须与构建机一致"]
        F2 --> F3["派生 Dockerfile：注入 yrsdk 运行时<br/>access_mode 含 tui 时追加 sshd 层"]
        F3 --> F4["docker build → tag name:version"]
        F4 --> F5["apply_runtime_spec：<br/>ports = access_mode 端口（tui 额外加 tcp:2222）<br/>rootfs.user = agentos，runtime = python3.11"]
    end

    F5 -->|"BuildResult：name / version / image_ref / runtime_spec / image_digest"| G["管理面轮询 probe_build 取结果"]
    G --> H["_register_payload 组装卡片记录：<br/>framework = name，description，access_mode，<br/>runtime_spec.rootfs.imageurl = image_ref<br/>状态 = registering"]
    H -->|"POST /api/images"| I{"注册中心返回"}
    I -->|"registered / updated"| J["状态 = registered<br/>卡片同时出现在用户视图与管理员视图"]
    I -->|"其它 / 报错"| K["状态 = register_failed<br/>在未注册列表可重试注册"]
    K -.->|"重试 _run_registration（复用已有 payload）"| H
    J --> L["PUT /api/images/{name}/default<br/>指定默认版本（用户视图按它启动）"]
    J --> M["删卡：先查实例，有实例则拒绝<br/>无实例才按卡片路径拆：本机文件 + 已 load 镜像 + 注册记录"]
```

**Tips**：上架的"成功"定义：`POST /api/images` 返回 `registered` / `updated`。构建产物（镜像、`runtime_spec`）只是注册的输入；管理面本地的 `LocalAgentPackage` 只是构建与注册的台账，卡片本身不落管理面库（卡片信息必须来自注册中心,保证数据唯一性）。

### 3.2 上架时序图

```mermaid
sequenceDiagram
    autonumber
    actor Admin as 管理员
    participant B as 浏览器
    participant CP as 管理面 :8090
    participant F as 镜像工厂 :8091
    participant R as 注册中心 :4003

    Admin->>B: 选择制品 + 填写 name / description / access_mode
    B->>CP: POST /api/v1/thirdparty_agent/cards（admin 鉴权）
    CP->>CP: 通用门禁（大小 / 命名）+ 落盘 packages/{digest}.artifact
    CP->>CP: 建 BuildTask + 绑定 launch sidecar 到制品
    CP-->>B: 202 {digest, request_id}
    CP->>F: build_from_path(path, options={access_mode, inject_ssh, registration_name})
    F->>F: 选 Recipe → 解析制品 → 平台校验 → 派生 Dockerfile
    F->>F: docker build → apply_runtime_spec
    F-->>CP: BuildResult(name, version, image_ref, runtime_spec, image_digest)
    loop 每 1 秒
        CP->>F: probe_build(request_id)
        F-->>CP: progress / 最终状态
    end
    CP->>CP: _register_payload 组装卡片记录（状态 = registering）
    CP->>R: POST /api/images
    R-->>CP: status = registered / updated
    CP->>CP: 状态 = registered，BuildTask 标记完成
    B->>CP: GET /cards（刷新列表）
    CP->>R: GET /api/images
    R-->>B: 卡片列表（名称 / 版本 / 描述 / access_mode）
```

### 3.3 制品状态机

```mermaid
stateDiagram-v2
    [*] --> uploaded: 通用门禁通过并落盘
    uploaded --> building: 异步构建任务启动
    building --> registered: 构建成功且注册成功
    building --> build_failed: 构建失败
    building --> register_failed: 构建成功、注册失败
    build_failed --> building: 重试构建
    register_failed --> registered: 重试注册（复用已存 payload）
    registered --> [*]: 删卡（需先确认无实例）
    note right of registered
        只有 registered 才成为卡片；
        其它状态只出现在"未注册"列表，
        不进用户视图
    end note
```

### 3.4 耦合点与变动点分析

> 判断规则：把这一侧换成另一个人写或另一家产品，对面要不要改代码——不用改 = 约定（显式化 + 校验点），要改 = 耦合（抽接口），根本换不了 = 未抽象（补扩展点）；单进程内单一 owner 的高频点三类都不算。

**耦合点：换实现必须改对面**

| # | 位置 | 耦合双方 | 现状依赖的实现细节 | 风险爆发点 |
|---|---|---|---|---|
| C1 | 制品身份 = `{digest}.artifact` 的文件名 | 管理面 ↔ 卡片 `package_path` | 注册中心不认识 digest，管理面用正则 `([0-9a-f]{64})\.artifact` 从卡片路径反解 | 路径或命名规则一改，反解静默返回 None——已注册判重失效、删卡找不到本地台账 |
| C2 | 本机路径 `package_path` / `image_archive_path` | 管理面 → 卡片 | 卡片里存了管理面的文件系统布局（launch sidecar 也按它定位） | 管理面多副本或迁移后，卡片找得到、sidecar 读不到 |
| C3 | 删卡级联 | 管理面 → 注册中心 → 工厂 | 固定顺序、无补偿：查实例 → 升默认版本 → 删注册记录 → 删本机包与归档 → 工厂卸镜像 | 中途失败留孤儿（记录没了镜像还在，或反之），无回滚 |

**约定：换实现不必改对面，但现在只是默契**

| # | 约定 | 参与方 | 现状 | 显式化手段 |
|---|---|---|---|---|
| A1 | `name` = `framework` | 管理面写卡 → Gateway 消费 | 写入时强制同值，无校验点 | 落卡时加一致性校验，契约里写明三者同源 |
| A2 | `runtime_spec` 字段语义 | 工厂 → 卡片 → Gateway | JSON blob 透传（工厂改 `rootfs.user/ports`，管理面只覆盖 `rootfs.imageurl`），无版本、无必需字段校验 | 声明版本 + 落卡前校验必需字段 |

**重复实现：同一语义多处各写一遍**

| # | 语义 | 现状写在哪几处 | 处理方向 |
|---|---|---|---|
| R1 | `access_mode` 的取值语义（`web` 取端口与启动命令、`tui` 触发 sshd） | 管理面 `_inject_ssh` / `_DEFAULT_ACCESS_PORTS`、工厂 `access_mode_ports`、Gateway `http_web_port_from_access_mode` / `_with_web_access_cmds`（消费端见 §4.4 R2） | 语义集中到单一解析器；新增取值时消费端显式失败，而不是静默降级 |

**变动点**

| # | 变动点 | 驱动力 | 现状 | 影响半径 |
|---|---|---|---|---|
| V1 | `access_mode` 取值集合 | 产品需求（新 agent 形态） | 无集中解析器（= R1） | 管理面 + 工厂 + Gateway |
| V2 | Recipe 集合 | 产品需求（新包类型） | `RecipeRegistry` 插件化 | 仅工厂 —— 已解耦，不必再动 |
| V3 | 注入层 `yrsdk`、`inject_ssh`、`_DEFAULT_RUNTIME=python3.11` | 上游底座升级 | 工厂内 Dockerfile 片段 | 仅工厂，但产物语义影响 Gateway 探针 |
| V4 | `runtime_spec` 语义 | 上游 openYuanRong | 无版本协商（= A2） | 工厂 + Gateway |
| V5 | 卡片字段集与 sidecar schema | 上游 a2x-registry | sidecar 已 v1 → v2 → v3 | 三方 |
| V6 | 管理面本地台账状态机 | 自身实现 | 单进程单 owner | 仅管理面 —— 三类都不算 |

**优先级与处理方向**

| 优先级 | 项 | 处理方向 |
|---|---|---|
| P0 | R1 / V1 access_mode | 语义集中到单一解析器 + 契约测试；新增取值时消费端显式失败 |
| P0 | C1 制品身份 | digest 升为卡片一等字段（或独立制品注册表），禁止从路径反解 |
| P1 | A2 / V4 runtime_spec | 声明为带版本的外部契约，落卡前校验必需字段 |
| P1 | C2 本机路径 | 卡片只存逻辑 locator，物理路径由管理面内部解析 |
| P1 | C3 删卡级联 | 定义幂等、可重试的拆除契约，补齐补偿 |
| P2 | A1 name / framework | 落卡时加一致性校验，成本低 |
| 不动 | V2 Recipe、V6 台账 | 已解耦 / 单侧自持 |

## 4. 图二：从管理面开始的启动跳转

### 4.1 跳转流程图

```mermaid
flowchart TD
    A["用户：个人智能体页点击启动"] --> B["GET /cards/{name}/launch-settings"]
    B --> C{"launchable?"}
    C -->|"无默认版本 / 非 web access_mode / 未配置 Gateway 地址"| C1["展示原因，不跳转"]
    C -->|"模板要求 MODEL_API_KEY 且用户未配置"| C2["引导先配置模型 Key"]
    C -->|"可启动"| D["进入启动页 /personal/{name}/launching"]
    D --> E["隐藏 iframe 提交 POST /cards/{name}/launch"]
    E --> F["管理面：渲染 launch 模板写入用户 home<br/>303 → http://{浏览器可达 host}:19000/{agent}/?user_id=&token="]
    F --> G["iframe 内请求 Gateway<br/>（首次请求会触发建实例，冷启动 502 藏在 iframe 里）"]
    D --> H["启动页并行轮询 GET /cards/{name}/launch-status<br/>间隔 1s，最长 30s"]
    H --> I{"注册中心实例 status = 运行?"}
    I -->|"是"| J["主窗口再次 POST /cards/{name}/launch → 303 → 真实进入智能体"]
    I -->|"超时 / 查询异常"| K["兜底：主窗口直接跳转（fallback）"]
```

**这张图的读法**：管理面**不创建实例**，只做两件事——把用户配置写好、把浏览器引到 Gateway。实例由 Gateway 在**首次请求**时按需创建；启动页的 iframe 是先遣队（把冷启动的等待与 502 关在 1px iframe 里），`launch-status` 只读注册中心、**不会触发创建**，等到实例 `运行` 后再让主窗口真正跳转。

### 4.2 启动跳转时序图

```mermaid
sequenceDiagram
    autonumber
    participant B as 浏览器 / 启动页
    participant M as 管理面 :8090
    participant G as Gateway :19000 WebProxy
    participant IAM as IAM
    participant R as 注册中心 :4003
    participant Y as YuanRong Frontend :8888
    participant S as 实例内 Web 服务

    B->>M: POST /cards/{name}/launch（同源表单，自动带 Cookie）
    M->>M: 校验默认版本 + web access_mode + 推导 Gateway origin
    M->>M: 渲染 launch 模板 → 原子写入 /home/agentos/users/{user}/
    M-->>B: 303 Location: http://host:19000/{agent}/?user_id={user}&token={JWT}

    B->>G: GET /{agent}/?user_id=&token=
    G->>IAM: POST /api/v1/auth/verify {token}
    IAM-->>G: data.valid / user_id / username
    G->>G: 门禁只认 valid；声称的 user_id 必须等于 IAM 的 user_id 或 username

    G->>G: 查内存 AgentManager（user_id + agent_type）
    alt 内存中无 READY runtime
        G->>R: GET /api/images/{framework}/launch-spec
        R-->>G: runtime_spec + env_vars + access_mode
        G->>Y: POST /api/agent（inline runtime_spec、workspace、env_vars）
        Y-->>G: instance_id
        G->>G: runtime 置 READY（创建返回不等注册完成）
        G->>R: POST /api/instances（address 先占位 = instance_id）
        G->>Y: GET /api/agent/{id} 轮询到 status = running
        G->>R: PATCH /api/instances/{service_id}（回填 node_ip / sandbox_ip）
        G->>Y: 南向探测（探针端口 = access_mode.web.port）
    end

    G->>Y: GET /serverless/v1/http?instance={id}&tenant_id=default&port={web_port}
    Y->>S: 转发到实例内端口
    S-->>Y: HTML / 响应头
    Y-->>G: 上游响应
    G-->>B: 剥掉 Cookie / Authorization / X-Token，注入 X-Forwarded-User 后回传

    Note over B,M: 同时，启动页每隔 1s 观察就绪
    B->>M: GET /cards/{name}/launch-status
    M->>R: GET /api/instances?framework=&kind=三方&user=
    R-->>M: 实例列表
    M-->>B: ready = true（该用户该框架存在 status = 运行 的实例）
    B->>M: POST /cards/{name}/launch（主窗口 _self）
    M-->>B: 303 → 主窗口进入智能体（此时 runtime 已 READY，直接反代）
```

### 4.3 access_mode 决定跳转方式

| 卡片 `access_mode` | 入口 | 管理面动作 | Gateway 动作 |
|---|---|---|---|
| `web`（如 OpenClaw / DSH 的 Web 服务） | 浏览器 `:19000/{agent}/` | 303 跳转 +（可选）渲染配置 | 建/复用实例 → HTTP/WS 反代到 `access_mode.web.port` |
| `tui`（如 Claude Code 等终端形态） | TUI：`3rdagent.switch` 后 SSH `-p 2222` | 不涉及浏览器跳转 | 建实例（镜像含 sshd 层）→ 返回 `ssh_ip` / `ssh_port` + 一次性私钥 |

本文只展开 `web` 形态的跳转，`tui` / SSH 的建立过程另文描述。

### 4.4 耦合点与变动点分析

> 判断规则同 §3.4。本链路额外一类是**时序耦合**（一侧的真实时序变成对面的依赖）；身份相关内容属"未抽象"，见 §5。

**耦合点：换实现必须改对面**

| # | 位置 | 耦合双方 | 现状依赖的实现细节 | 风险爆发点 |
|---|---|---|---|---|
| C4 | `service_id = generic_{sha256(user\0framework)[:8]}` | Gateway 派生 → 注册中心 → 管理面反查 | 管理面按 `framework` + `kind` + `user` 查实例、自己判就绪 | 派生算法一改，历史实例被判成"不是我的"：就绪判定与删卡闸门同时失效 |
| C5 | 实例就绪的判定链 | Gateway → 注册中心 → 管理面 | Gateway 建沙箱后**异步**注册（先占位后 PATCH），管理面靠轮询注册中心判断就绪 | 注册失败或延迟时，管理面判"未就绪"而实例其实可用（或反之），两侧无对账 |

**约定：换实现不必改对面，但现在只是默契**

| # | 约定 | 参与方 | 现状 | 显式化手段 |
|---|---|---|---|---|
| A3 | 实例 `kind` 与状态词汇（`三方` / `运行` / `异常`） | 管理面 ↔ 注册中心 ↔ Gateway | 三处硬编码同一批中文字符串 | 共同定义枚举，双方引用同一事实源 |
| A4 | 跳转 URL 与端口（`{origin}/{agent}/?user_id=&token=`，`AGENTBOX_WEB_PORT=19000`） | 管理面 → Gateway | 两侧常量与注释对齐，端口硬编码 | 端口与参数纳入部署配置，写进契约测试 |
| A5 | launch 模板的占位符与 `relative_path` 形状 | 管理面渲染 → agent 镜像消费 | 白名单只约束占位符，模板内容与 agent 实际配置结构的一致性无校验 | 校验模板必需键与路径，或提供样例校验 |

**重复实现：同一语义多处各写一遍**

| # | 语义 | 现状写在哪几处 | 处理方向 |
|---|---|---|---|
| R2 | `access_mode` 的运行时解释 | Gateway `http_web_port_from_access_mode`、`_with_web_access_cmds`、`_third_agent_probe_port` / `resolve_web_endpoint`（生产端见 §3.4 R1） | 消费端遇到未知取值显式失败；解释逻辑与 §3.4 R1 收口到同一处 |

**变动点**

| # | 变动点 | 驱动力 | 现状 | 影响半径 |
|---|---|---|---|---|
| V7 | 实例注册协议与状态词汇 | 上游 a2x-registry | 无版本协商（= A3） | Gateway + 管理面 |
| V8 | Gateway 的 registry 客户端与 launch-spec 解析 | 上游 jiuwenswarm | 随包升级 | Gateway 单侧，契约另一头是注册中心 |
| V9 | YR `/api/agent`、`/serverless/v1/*` 与实例状态机 | 上游 openYuanRong | 无版本协商 | 工厂 + Gateway |
| V10 | 前端启动预热与轮询策略（iframe 预热、1s / 30s） | 交互与浏览器兼容 | 前端自持 | 仅前端 —— 三类都不算 |

**优先级与处理方向**

| 优先级 | 项 | 处理方向 |
|---|---|---|
| P0 | C4 实例查找 | 收口为"按用户 + 框架查实例状态"的接口，派生规则不再被外部依赖 |
| P0 | C5 就绪判定 | 明确就绪的唯一来源与重试、对账机制，去掉"是否已注册"的隐式依赖 |
| P1 | A3 / V7 状态与 kind 词汇 | 与注册中心共同定义枚举，双方引用同一事实源 |
| P1 | A4 跳转 URL | 端口与参数纳入部署配置，写进契约测试 |
| P1 | V8 / V9 上游依赖 | 用契约测试锁住依赖字段，跟随升级并保留兼容窗口 |
| P2 | A5 模板一致性 | 校验模板必需键与路径 |
| P2 | R2 access_mode 消费端 | 与 §3.4 R1 同一契约；未知取值显式失败 |
| 不动 | V10 前端策略 | 单侧自持 |

## 5. 未抽象能力：身份与鉴权

> 判断规则（未抽象）：能力没有替换路径——没有扩展点、没有策略选择，第三方实现无法接入。这类不是耦合，是缺 seam。

**一个用户名字符串同时承担四个角色**

| 角色 | 落地位置 | 换用户系统时的后果 |
|---|---|---|
| 会话身份 | JWT subject / `username` | 客户 IdP 的主键通常不是稳定用户名 |
| 工作区路径 | `/home/agentos/users/{username}`：管理面 `home_path()` + `AGENTOS_HOME_BASE`，Gateway `resolve_agent_workspace` + `DEFAULT_AGENT_WORKSPACE_ROOT` | 用户名变化或含非法字符时，目录与实例数据错位 |
| 实例归属 | 注册中心实例行 `user` + Gateway `service_id` 的 `user` 段 | 归属键随用户名字符串漂移 |
| SSH 授权主体 | `KeyRegistryEntry.username` 必须等于 SSH 登录名 | 客户用户系统无此约定即无法授权 |

**两侧的抽象程度不同**

| 侧 | 用户存储 | 凭据形态与验证 | 证据 |
|---|---|---|---|
| 管理面 | **有抽象**：`AbstractUserBackend` + `settings.USER_SYSTEM_BACKEND` | **无抽象**：对称 JWT（`AGENTOS_JWT_SECRET_KEY` + 可配算法）本地签发与验签，`raw_token` 原样透传给 Gateway | `app/services/base.py`、`iam/tokens.py`、`iam/security.py` |
| Gateway | 无（不持有用户） | **无抽象**：`AgentOSAuthenticator` 直接 new；`ExtensionRegistry` 只有 `agent_server_client` / `crypto_utility` / `third_agent` 三个注册口，**没有认证口**；SSH 侧 `SshPublicKeyAuthenticator` 同样硬编码 | `extensions/agentos/agentos_router/extension.py`、`extensions/registry.py`、`gateway/channel_manager/protocol/ssh/server.py` |

**契约被固化成的具体形态**：`POST {auth_service_url}/api/v1/auth/verify`，body `{token, resource_id: "apps", action_id: "read"}`，且只看 `data.valid`。默认地址 `http://iam-agentos:8090`；管理面代码里没有这个端点，验证方是另一个服务，必须与签发方共享密钥与用户口径——这个假设没有任何地方声明过。默认 `agentos.auth_enabled: false`，此时 Gateway 回落读客户端声明的 `X-User-Id` / `?user_id=`，不做任何校验。

**对照：平台已有的 seam**（说明缺的是这一块能力，而不是平台没有扩展机制）

- 接入通道：`ChannelManager.register_channel`（web / tui / im / ssh / web_proxy 都是注册进来的）
- 扩展：`ExtensionRegistry.register_third_agent` / `register_agent_server_client` / `register_crypto_utility`
- 构建方式：`RecipeRegistry`
- 用户存储：`AbstractUserBackend` + `USER_SYSTEM_BACKEND`
- 半成品：`SshKeyIssuer` 只有 Protocol、没有注册口

**结论**：新增一种接入协议能接，客户自带用户系统接不了。要接入至少补三件事——① 身份验证的扩展点与策略选择；② 把上述四个角色拆开（identity subject / workspace owner / instance tenant / ssh principal）；③ 与客户用户系统约定稳定的主体标识。

## 6. 依据与联动

- 上架：`backend/app/api/v1/thirdparty_agent.py`（`/uploads*`、`/cards*`）、`services/thirdparty_agent_service.py`（`publish*`、`_publish_accepted`、`_run_build`、`_register_payload`、`_normalize_access_mode`）、`services/image_process_client.py`、`sandbox-manager/image_process/app/factory/{service,recipe,inject}.py`、`app/factory/recipes/oci_archive.py`
- 跳转：`thirdparty_agent/launch_config.py`（`gateway_origin`、`render_launch_config`、`env_path_for`）、`frontend/src/views/resources/agent/PersonalAgentLaunchingPage.vue`、`frontend/src/api/framework.ts`（`submitLaunchForm`）
- Gateway：`jiuwenswarm 0.2.4b4` 的 `web_proxy_connect.py`（`_authenticate_web_proxy`、`_forward_headers`）、`agentos_router/router_client.py`（`resolve_web_endpoint`、`_create_agent`、`_register_agent`、`thirdagent_switch`）、`yuanrong_frontend_client.py`

**图需要随代码更新而更新的触发点**：`access_mode` 的生产与消费语义（§3.4 R1 / §4.3 / §4.4 R2）、实例身份与就绪判据（§4.4 C4 / C5）、跳转端点契约（§4.2 / §4.4 A4）、身份与鉴权（§5）。
