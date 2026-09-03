---
title: "三方 Agent 新旧数据库对比与变更计划"
type: design
domain: agent
status: draft
---

# 三方 Agent 新旧数据库对比与变更计划

> 文档状态：草案  
> 适用范围：AgentOS Control Panel 三方智能体管理、Agent 注册中心  
> 代码基线：管理面 `master` 分支  
> 注册中心契约：`registry_openapi.yaml`，OpenAPI 0.1.0  
> 关联设计：`third-party-agent-artifact-image-factory-design-v2.md`

## 1. 结论

变更后的注册中心已经能够承接卡片的当前状态、展示信息和启动信息，可以成为三方智能体卡片的唯一事实源。因此管理面的旧数据库需要做变更，不再需要loacl merge构建卡片。

目标数据边界为：

1. 注册中心保存完整卡片，包括 `name + version` 主键、描述、默认版本、运行规格、`access_mode`、本机包路径和镜像归档路径等。
2. `access_mode` 是原单一 `launch_command` 的结构化升级；TUI 是默认接入方式，默认端口为 `2222`，其 `cmd` 保存原启动命令。Web 接入方式额外保存端口和对应命令。
3. 管理面本机只保存尚未注册成功的软件包和构建历史。
4. 当前 `agent_registrations` 在迁移完成后删除；当前 `build_tasks` 拆分或演进为未注册包状态与构建历史，不再参与卡片投影。

```text
注册中心
└─ ImageEntry：当前卡片唯一事实源
   ├─ 展示字段
   ├─ 默认版本
   ├─ access_mode
   ├─ runtime_spec
   └─ 本机资源定位信息

管理面本机数据库
├─ local_agent_packages：仅未注册软件包
└─ agent_build_history：每次构建/注册尝试历史
```

## 2. 当前数据模型

### 2.1 `agent_registrations`

| 字段 | 当前语义 |
|---|---|
| `framework` | 旧注册中心的框架标识；兼容期实际承载管理员填写的启动命令 |
| `framework_version` | 版本，与 `framework` 组成联合主键 |
| `installer_path` | 本机源软件包路径 |
| `agent_name` | 镜像工厂解析出的制品名称 |
| `display_name` | 本地展示名称 |
| `created_at` | 本地记录创建时间 |

该表当前用于补齐注册中心未返回的 `package_path` 和 `agent_name`。卡片列表和详情因此不是单一来源，而是注册中心数据与本地记录的合并结果。

### 2.2 `build_tasks`

| 字段 | 当前语义 |
|---|---|
| `task_id` | 构建请求 ID |
| `installer_path` | 输入软件包路径，同时被用于关联未注册包 |
| `status` | `pending/building/done/failed` 等任务状态 |
| `progress` | 构建进度 |
| `image` | 构建产出的镜像引用 |
| `image_digest` | 镜像摘要 |
| `created_at` | 任务创建时间 |
| `started_at` | 开始时间 |
| `finished_at` | 结束时间 |
| `error_message` | 失败信息 |

该表同时承担任务状态、并发锁、构建历史和未注册包索引。当前“未注册包”不是显式实体，而是通过 `build_tasks.installer_path - agent_registrations.installer_path` 的差集推导。

## 3. 变更后注册中心模型

新版 `ImageEntry` 以 `name + version` 标识一条卡片版本，`framework` 降级为普通展示字段。

| 分类 | 字段 | 用途 |
|---|---|---|
| 身份 | `name`、`version` | 稳定定位一个卡片版本 |
| 展示 | `framework`、`description` | 卡片展示与筛选 |
| 版本 | `is_default` | 标识同一 `name` 的默认版本 |
| 接入 | `access_mode[]` | 保存 TUI/Web 等访问方式、端口和启动命令 |
| 运行 | `runtime_spec`、`env_vars`、`workspace`、`mounts` | 拉起实例所需规格 |
| 镜像 | `image_module_version`、`runtime_spec.rootfs.imageurl` | 镜像模块版本和镜像引用 |
| 本机资源 | `package_path`、`image_archive_path` | 管理员详情与整卡清理 |
| 审计 | `uploaded_by`、`created_at` | 上传人与创建时间 |

注册中心同时提供：

- `GET /api/images`：卡片版本列表；
- `POST /api/images`：注册或更新卡片；
- `PATCH /api/images/{name}/{version}`：更新可变字段；
- `DELETE /api/images/{name}/{version}`：校验实例占用后注销卡片；
- `PUT /api/images/{name}/default`：设置默认版本；
- `GET /api/images/{name}/launch-spec`：提供实例拉起规格；
- `GET /api/instances`：提供实例状态及卡片实例统计来源。

### 3.1 `access_mode` 约定

`access_mode` 与原 `launch_command` 是包含关系，不再新增重复的卡片级启动命令字段。

```json
{
  "access_mode": [
    {
      "name": "tui",
      "port": "2222",
      "cmd": "opencode"
    },
    {
      "name": "web",
      "port": "4096",
      "cmd": "opencode gateway --port 4096"
    }
  ]
}
```

约束如下：

1. TUI 为默认接入方式。
2. TUI 默认端口为字符串 `"2222"`。
3. 当前管理面单一 `launch_command` 输入映射为 `access_mode[name=tui].cmd`。
4. Web UI 为可选方式，同时保存前端展示所需的端口和启动命令。
5. 同一张卡内 `access_mode.name` 唯一；管理面不得依赖数组顺序查找 TUI。
6. `framework` 不再承载启动命令。当前混用仅是旧注册中心缺少结构化接入字段时的兼容措施。

### 3.2 `name` 获取与后续扩展

`name` 是注册中心卡片的稳定主键之一，不能把“所有制品都能自动解析名称”作为长期前提。

当前 OpenCode、Claude 等软件包可以从解包后的元数据中取得名称和版本，适合自动填入；后续支持 OCI 镜像、通用压缩包或自定义制品时，镜像 tag、文件名和内部标签可能缺失、可变或相互冲突，不能始终作为稳定的软件身份。

名称来源按以下优先级处理：

1. 制品内受支持且通过校验的标准元数据，例如包清单中的 `name/version`。
2. 镜像内约定的 OCI annotation 或 AgentOS manifest。
3. OCI archive 中唯一且合法的 repository tag，只能作为候选值，不能在存在多个 tag 时自行选择。
4. 无可靠元数据或存在歧义时，由管理员填写或确认 `name/version`。

无论名称来自自动解析还是管理员输入，最终都必须经过统一规范化和合法性校验。

该问题不阻塞本次数据库迁移。当前支持的软件包能够由镜像工厂在 `buildFromPath` 过程中解析 `name/version`；管理员在上传请求中已经同时提交完整 `access_mode`，因此影响构建的配置在构建开始前已经确定。工厂完成构建并返回名称、版本后，管理面再调用注册中心，当前时序能够满足要求：

```text
上传文件 + access_mode
→ 通用门禁、计算摘要并落盘
→ buildFromPath(path, access_mode/options)
→ 工厂解析并返回 name/version 和构建结果
→ 注册中心登记
```

未来扩展到无法稳定自动取名的 OCI 镜像或其他制品后，再增加 `inspectFromPath/preflight`，把“识别名称”和“完整构建”拆开。该能力放在迁移计划最后实施，不作为新旧数据库切换的前置条件。

## 4. 目标本机数据库

### 4.1 `local_agent_packages`

只保存尚未成功注册的软件包。注册中心确认写入成功后删除该记录。

| 字段 | 类型建议 | 说明 |
|---|---|---|
| `content_digest` | `CHAR(64) PK` | 源包 SHA-256，也是去重和锁键 |
| `package_path` | `VARCHAR(1024) UNIQUE NOT NULL` | 服务端生成并校验归属的本机路径 |
| `original_filename` | `VARCHAR(512) NOT NULL` | 原始文件名，仅用于展示 |
| `size_bytes` | `BIGINT NOT NULL` | 实际落盘大小 |
| `uploaded_by` | `VARCHAR(255) NOT NULL` | 上传管理员 |
| `access_mode` | `JSON NOT NULL` | 上传时提交的接入配置，至少包含默认 TUI |
| `state` | `VARCHAR(32) NOT NULL` | `uploaded/building/build_failed/registering/register_failed` |
| `locked_by` | `VARCHAR(64) NULL` | 当前持锁请求 ID |
| `locked_until` | `TIMESTAMPTZ NULL` | 锁超时时间 |
| `last_error_code` | `VARCHAR(64) NULL` | 结构化错误码 |
| `last_error_message` | `TEXT NULL` | 最近一次失败原因 |
| `created_at` | `TIMESTAMPTZ NOT NULL` | 创建时间 |
| `updated_at` | `TIMESTAMPTZ NOT NULL` | 更新时间 |

这里保存 `access_mode` 而不是单独保存 `launch_command`，避免重试时丢失 TUI/Web 配置，也与注册中心请求模型保持一致。整体变更模式也是显式传入，TUI和Web会在前端传过来，只有TUI端口需要缺省为2222。

### 4.2 `agent_build_history`

每次构建或注册尝试保存一条历史，只用于进度、审计和排障，禁止参与当前卡片投影。

| 字段 | 类型建议 | 说明 |
|---|---|---|
| `request_id` | `VARCHAR(64) PK` | 一次尝试的唯一标识 |
| `content_digest` | `CHAR(64) NOT NULL` | 源包摘要；建索引，不强依赖未注册包外键 |
| `package_path_snapshot` | `VARCHAR(1024) NULL` | 当时路径快照，文件可能已被清理 |
| `operation` | `VARCHAR(16) NOT NULL` | `publish/retry/register_retry` |
| `phase` | `VARCHAR(32) NOT NULL` | `queued/building/registering/completed/failed/cancelled` |
| `progress` | `INTEGER NOT NULL` | 0～100 |
| `uploaded_by` | `VARCHAR(255) NOT NULL` | 发起人 |
| `access_mode_snapshot` | `JSON NOT NULL` | 当次注册使用的接入配置 |
| `artifact_name` | `VARCHAR(256) NULL` | 工厂解析出的名称 |
| `artifact_version` | `VARCHAR(128) NULL` | 工厂解析出的版本 |
| `image_ref` | `VARCHAR(512) NULL` | 镜像引用 |
| `image_digest` | `VARCHAR(128) NULL` | 镜像摘要 |
| `image_archive_path` | `VARCHAR(1024) NULL` | 镜像归档路径 |
| `recipe_id` | `VARCHAR(128) NULL` | 构建策略追溯 |
| `base_ref` | `VARCHAR(512) NULL` | 基础镜像追溯 |
| `registry_name` | `VARCHAR(256) NULL` | 注册成功后的卡片主键快照 |
| `registry_version` | `VARCHAR(128) NULL` | 注册成功后的版本快照 |
| `error_stage` | `VARCHAR(32) NULL` | `gate/factory/registry/cleanup` |
| `error_code` | `VARCHAR(64) NULL` | 结构化错误码 |
| `error_message` | `TEXT NULL` | 失败详情 |
| `created_at` | `TIMESTAMPTZ NOT NULL` | 创建时间 |
| `started_at` | `TIMESTAMPTZ NULL` | 开始时间 |
| `finished_at` | `TIMESTAMPTZ NULL` | 结束时间 |

构建追溯字段可以只保存在本机构建历史中，不要求注册中心同时保存；注册中心负责当前卡片，构建历史负责过去发生过什么。

## 5. 变更后的业务流程

### 5.1 上架

```text
管理员上传软件包并填写 access_mode
→ 管理面通用门禁、计算 content_digest、落盘
→ 创建 local_agent_packages
→ 创建 agent_build_history
→ 按摘要原子加锁
→ 镜像工厂 buildFromPath(path, access_mode/options)
→ 工厂解析并返回 name/version 和构建结果
→ POST 注册中心 /api/images
→ 回读确认 name + version 已存在
→ 构建历史置 completed
→ 删除 local_agent_packages
→ 释放锁
```

当前 OpenCode、Claude 等受支持软件包的 `name/version` 由工厂在构建过程中自动解析，无需增加预检接口。若工厂构建成功但注册中心失败，软件包保留为 `register_failed`。重试时优先只重试注册，不应无条件重复执行镜像构建。

### 5.2 卡片查询

```text
GET 管理面 cards
→ GET 注册中心 /api/images
→ 按用户角色裁剪字段
→ 管理员视图按需查询 /api/instances 并聚合实例数
→ 返回前端
```

查询过程中不读取 `local_agent_packages` 或 `agent_build_history` 补齐卡片字段。实例接口失败时返回错误或未知状态，禁止投影为零。

### 5.3 未注册包查询与重试

未注册包列表直接查询 `local_agent_packages`。重试沿用其中保存的完整 `access_mode`；前端可以允许管理员修改后再发起新尝试，每次尝试新增一条构建历史。

### 5.4 卡片删除

```text
管理面请求删除 name + version
→ 注册中心校验是否存在关联实例
→ 有实例：409，所有资源保持不变
→ 无实例：进入受控清理流程
→ 清理本机软件包、镜像归档和已加载镜像
→ 注销注册中心卡片
```

管理面只能删除配置目录下经过归属校验的路径，不能直接信任注册中心返回的任意绝对路径。跨组件删除不是数据库事务，应记录清理阶段并提供幂等重试或定期对账。

默认版本的选择、删除后提升等规则由注册中心保证，管理面和前端不再分别计算最高版本。

## 6. 数据迁移计划

迁移按照“先适配已发布的注册中心契约，再建立本机新表、迁移数据、切换读取，最后下线旧表”的顺序实施。名称预检属于后续制品扩展，不阻塞数据库迁移。

### 6.1 阶段 0：确认已发布的注册中心契约

1. 固定卡片主键为 `name + version`，`framework` 仅作为展示或分类字段。
2. 固定 `access_mode` 结构及唯一性规则；TUI 默认 `name=tui`、`port=2222`。
3. 默认版本接口统一使用新版字段 `version`。
4. 明确 `PATCH` 对 `access_mode`、`runtime_spec` 等复合字段采用整体替换还是局部合并。
5. 明确 `package_path`、`image_archive_path` 的返回范围和本机路径校验责任。
6. 确认当前受支持软件包继续由 `buildFromPath` 自动解析并返回 `name/version`。

阶段完成标准：管理面可以只使用新版注册中心字段完成卡片注册、查询、修改、默认版本切换和删除。

### 6.2 阶段 1：升级注册中心客户端和卡片 DTO

1. `list/get/delete/default` 全部切换为 `name + version`。
2. `version` 为标准字段；过渡期只在客户端适配层回退读取 `framework_version`。
3. 新增 `patchImage`，描述等字段修改走 `PATCH`。
4. 发布请求发送 `name`、`version`、`framework` 和 `access_mode`，不再把命令写入 `framework`。
5. 卡片 DTO 完整投影 `name/version/framework/description/is_default/access_mode` 等角色允许字段。
6. 前端卡片 Key、详情路由和操作参数切换为 `name + version`。
7. 修复卡片分页总数和单卡实例总数复用同一变量的问题。
8. 上传界面在发起构建前提交完整 `access_mode`；工厂构建结果返回 `name/version` 后再注册。

本阶段保留旧数据读取兼容，但所有新写入只使用新版协议。

### 6.3 阶段 2：创建新本机表并切换新任务写入

1. 创建 `local_agent_packages`，显式保存未注册包、`access_mode`、当前状态和摘要锁。
2. 创建 `agent_build_history`，保存每次构建和注册尝试。
3. 对 `content_digest`、`package_path`、`request_id` 和常用状态查询建立约束或索引。
4. 使用数据库条件更新或行锁实现摘要互斥，不能只依赖进程内状态。
5. 增加超时锁回收和服务重启后的中断任务修复逻辑。
6. 新上传与重试写入新表；迁移观察期内可短期兼容写入 `build_tasks`。
7. 构建成功、注册失败时保留构建结果，支持只重试注册。

本阶段不改变卡片读取来源，也不删除旧表。

### 6.4 阶段 3：迁移存量数据

先迁移构建任务：

1. 每条 `build_tasks` 转换为一条 `agent_build_history`。
2. 根据任务状态映射新 `phase`，保留镜像引用、摘要、错误和时间字段。
3. 按路径计算或恢复 `content_digest`；无法恢复时记录迁移异常，不伪造摘要。

再迁移注册与未注册状态：

1. 逐条读取 `agent_registrations`，结合注册中心现有记录和构建历史确定 `name + version`。
2. 将旧 `framework` 中兼容承载的命令迁入 `access_mode[name=tui].cmd`，端口补为 `2222`。
3. 将 `installer_path` 回填为注册中心 `package_path`，补齐可确认的描述、归档路径和运行信息。
4. 写入注册中心后回读校验，不立即删除本地旧记录。
5. 注册中心不存在对应卡片且本机文件仍存在时，生成 `local_agent_packages`。
6. 注册中心已有卡片时只保留构建历史，不生成未注册包。
7. 本机文件不存在时只保留历史，并将资源缺失写入迁移报告。

必须单列人工处理项：名称无法确定、同名同版本冲突、本地路径丢失、旧 `framework` 无法确认是否为命令。

### 6.5 阶段 4：双读校验与读取切换

1. 在不影响用户响应的情况下，对比注册中心卡片与旧拼接卡片的数量、主键和关键字段。
2. 对差异分类：协议映射差异、存量脏数据、路径缺失、注册中心缺失和旧表孤儿记录。
3. 差异收敛后，卡片列表和详情只读取注册中心。
4. 删除 `_merge_local_card()` 及卡片查询中的 `AgentRegistration` 读取。
5. 未注册列表只读取 `local_agent_packages`。
6. 构建进度和历史只读取 `agent_build_history`。
7. 实例接口失败时返回明确错误或未知状态，不投影为零。

### 6.6 阶段 5：停止旧表写入

1. 停止写入 `agent_registrations`。
2. 停止写入 `build_tasks`，所有新任务只进入新表。
3. 保留旧表只读观察一个发布周期。
4. 持续对账注册中心卡片、本地未注册包与构建历史，确认没有新差异。
5. 准备回滚开关：读取可短期退回兼容适配层，但禁止重新形成长期双写。

### 6.7 阶段 6：下线旧表与兼容代码

1. 导出 `agent_registrations` 和 `build_tasks` 备份。
2. 先将旧表改名隔离，确认运行代码不存在访问。
3. 删除旧 ORM、Service 合并逻辑、差集推导逻辑和对应兼容测试。
4. 经过约定观察期后删除旧表。
5. 删除 `framework_version`、旧 `framework` 命令回退等客户端兼容分支。

迁移完成标准：删除两张旧表后，卡片展示与启动、当前上架流程、未注册包重试、构建历史查询和卡片删除均正常工作。

### 6.8 阶段 7：扩展名称预检能力（后续）

本阶段仅在 OCI 等新制品无法稳定自动取得 `name/version` 时实施，不属于本次数据库切换的完成条件。

1. 为镜像工厂增加 `inspectFromPath` 或等价 preflight 接口，只做制品识别和最小元数据解析，不执行完整构建。
2. 返回 `artifact_kind`、候选 `name/version`、`name_source`、候选 Recipe、名称是否可编辑及构建所需配置。
3. 名称优先读取标准包元数据或约定的 OCI annotation；唯一 repository tag 只能作为候选值。
4. 无可靠名称或存在多个候选值时，由管理员在构建前确认或填写。
5. 可靠元数据产生的名称默认只读；若允许重命名，分别保存制品原始名称和注册卡片名称。
6. 届时再按需要为 `local_agent_packages` 增加 `detected_name`、`detected_version`、`name_source`、`confirmed_name`、`confirmed_version` 和 `artifact_kind` 等字段。

该阶段引入后，上架流程才演进为“上传并预检 → 确认名称和接入方式 → 构建 → 注册”。

## 7. 兼容策略

迁移窗口内在管理面客户端设置单一兼容层：

```text
name       ← 新版 name；旧版临时回退 agent_name/framework
version    ← 新版 version；旧版临时回退 framework_version
TUI cmd    ← 新版 access_mode[name=tui].cmd；旧版临时回退 framework
```

兼容回退只能存在于注册中心客户端适配层，不能扩散到数据库模型、Service 和前端。所有新写入只使用新版字段。

## 8. 验收标准

完成变更需满足：

1. 删除 `agent_registrations` 后，卡片列表、详情、描述、默认版本、启动和删除功能不受影响。
2. 前端卡片不再读取或依赖任何本地卡片字段。
3. 默认 TUI 可以从 `access_mode` 稳定解析，端口为 `2222`。
4. Web 接入方式能够同时展示端口并取得对应命令。
5. `framework` 不再保存启动命令。
6. 已注册卡片不会出现在未注册包列表。
7. 构建失败和注册失败都保留包、接入配置及错误信息，并可以重试。
8. 构建成功但注册失败时，可以只重试注册。
9. 同一摘要并发上传、删除或重试只能有一个操作获得锁。
10. 注册中心或实例接口不可用时，页面明确报错，不显示虚假的空卡片或零实例。
11. 注册中心分页总数与卡片实例数互不覆盖。
12. 删除和迁移操作具备审计记录、幂等重试和异常对账能力。
13. OpenCode、Claude 等当前支持的制品能够由工厂构建结果稳定返回 `name/version`。
14. `access_mode` 在完整构建开始前已经确定，并实际参与对应 Recipe 的构建参数生成。

后续启用名称预检能力时，追加以下验收项：

1. 自动识别结果能够展示 `name/version` 及名称来源。
2. OCI 等名称缺失或存在多个候选值的制品不会静默选名，管理员可在构建前确认或补充名称。

## 9. 最终表处置

| 当前表 | 最终处置 | 替代者 |
|---|---|---|
| `agent_registrations` | 迁移后删除 | 注册中心 `ImageEntry` |
| `build_tasks` | 迁移后删除或停止使用 | `local_agent_packages` + `agent_build_history` |

最终原则是：注册中心回答“现在有哪些卡片、如何展示和启动”；本机数据库回答“哪些包尚未注册成功、过去发生过哪些构建和注册尝试”。两类数据不再互相补齐或推导。
