---
title: "三方 Agent 上架服务数据库表结构（管理面现状）"
type: reference
domain: agent
status: active
---

# 三方 Agent 上架服务数据库表结构（管理面现状）

> 文档状态：现状记录，随代码更新  
> 适用范围：AgentBox-Manager 管理面后端「三方 Agent 上架」链路（上传 → 构建 → 注册）  
> 代码基线：`AgentBox-Manager`，`feature/thirdparty-access-mode-image`，`693a0fe`  
> 数据库：PostgreSQL 18（`AGENTOS_DATABASE_URL`，与 LiteLLM 共享 `agentos` 库）  
> 表定义：`backend/app/models/thirdparty_agent.py`（本文 DDL 由该 ORM 模型按 PostgreSQL 方言导出）  
> 关联文档：[三方 Agent 新旧数据库对比与变更计划](third-party-agent-database-comparison-and-migration-plan.md)、[第三方 Agent 接入架构：现状、协议边界与理想模型](third-party-agent-access-architecture.md)  

## 1. 数据边界

管理面本机只保存**未注册成功的包 + 构建历史**，卡片与实例的权威数据在 Agent 注册中心：

| 数据 | 存放位置 | 说明 |
|---|---|---|
| 卡片、实例、默认版本、运行规格 | 注册中心 | 唯一事实源，管理面通过 `image_process_client` 读写 |
| 软件包文件、注册状态、构建历史 | 管理面数据库（本文 4 张表） | 包身份以整文件 SHA-256 为主键 |
| 启动模板、模板修订号 | 包同目录 sidecar 文件 `{digest}.metadata.json` | 不落库，见 §11 |
| 用户模型 API Key | 用户家目录下的 `.env` 文件 | 不落库，见 §11 |

## 2. 表清单

| 表名 | ORM 类 | 中文名 | 行粒度 | 主键 | 状态列 |
|---|---|---|---|---|---|
| `thirdparty_upload_sessions` | `UploadSession` | 分片上传会话 | 一次上传 = 一行 | `upload_id` | `state` |
| `thirdparty_upload_parts` | `UploadPart` | 上传分片 | 一个分片 = 一行 | `upload_id` + `part_number` | 无 |
| `local_agent_packages` | `LocalAgentPackage` | 本地软件包与上架状态 | 一个包内容 = 一行 | `content_digest` | `state` |
| `build_tasks` | `BuildTask` | 镜像构建任务 | 一次构建尝试 = 一行 | `task_id` | `status` |

按生命周期分为三段：`thirdparty_upload_*` 只服务于上传阶段且会被清理；`local_agent_packages` 是上架状态的唯一载体，包上传后长期保留；`build_tasks` 是构建历史，同一路径可累积多行。

## 3. 表关系

```text
thirdparty_upload_sessions ──1:N──> thirdparty_upload_parts
   (upload_id)                FK, ON DELETE CASCADE
        │
        │ artifact_digest = content_digest      （逻辑关联，无外键）
        │ build_request_id = task_id            （逻辑关联，无外键）
        ▼
local_agent_packages ──────1:N──> build_tasks
   (content_digest)           package_path = installer_path（逻辑关联，无外键）
```

**只有一处物理外键**（分片 → 会话，级联删除），其余三处均为值对应关系：

| 关联 | 对应字段 | 建立时机 | 注意 |
|---|---|---|---|
| 上传会话 → 软件包 | `artifact_digest` = `content_digest` | 分片合并完成、发布受理后 | 同一文件重复上传命中同一包行 |
| 上传会话 → 构建任务 | `build_request_id` = `task_id` | 发布受理时写入 | 仅记录首次触发的构建 |
| 软件包 → 构建任务 | `package_path` = `installer_path` | 包落盘后 | 一次查询取该包的全部构建历史 |

因此**不能用级联删除管理生命周期**：删除软件包行需自行清理对应的 `build_tasks` 行与磁盘文件。

## 4. `thirdparty_upload_sessions` — 分片上传会话

一次断点续传上传的会话记录。分片全部校验通过并合并后，在**同一行**上回填制品三件套（`artifact_id` / `artifact_digest` / `artifact_path`），供发布接口按 artifact 取包。

只有 `uploading` 与 `failed` 两个状态可以继续传分片和发起合并（代码中的 `_ACTIVE_STATES`）；`completing` 已加锁，重复调用合并会返回冲突而非重复执行。

### 4.1 字段

| 字段 | 类型 | 可空 | 键/索引 | 应用层默认 | 说明 |
|---|---|---|---|---|---|
| `upload_id` | `VARCHAR(64)` | 否 | 主键 | — | 上传会话 ID，服务端生成 |
| `owner_user_id` | `VARCHAR(64)` | 否 | 索引 | — | 归属用户 ID，读取时校验以隔离多用户 |
| `uploaded_by` | `VARCHAR(255)` | 否 | — | — | 上传人用户名（展示与审计用） |
| `original_filename` | `VARCHAR(512)` | 否 | — | — | 客户端原始文件名，仅作展示 |
| `total_size` | `BIGINT` | 否 | — | — | 客户端声明的总字节数，应用层限制 ≤ 2 GiB |
| `chunk_size` | `INTEGER` | 否 | — | — | 分片大小，来自 `THIRDPARTY_AGENT_UPLOAD_CHUNK_BYTES`（64 MiB） |
| `total_parts` | `INTEGER` | 否 | — | — | 分片总数，由 `total_size` 与分片大小推导 |
| `state` | `VARCHAR(32)` | 否 | 索引 | `uploading` | 会话状态，见 §4.3 |
| `artifact_id` | `VARCHAR(64)` | 是 | 唯一 | — | 制品 ID，合并完成后回填；未完成时为 `NULL` |
| `artifact_digest` | `VARCHAR(64)` | 是 | 索引 | — | 整文件 SHA-256 十六进制串，与软件包主键同值 |
| `artifact_path` | `VARCHAR(1024)` | 是 | — | — | 包落盘绝对路径，形如 `.../packages/{digest}.artifact` |
| `build_request_id` | `VARCHAR(64)` | 是 | — | — | 发布受理后写入，对应 `build_tasks.task_id` |
| `last_error` | `TEXT` | 是 | — | — | 最近一次失败原因 |
| `expires_at` | `TIMESTAMPTZ` | 否 | 索引 | — | 过期时刻 = 当前时刻 + TTL（3600 秒）；进入 `completing` 时刷新一次，避免合并期间被清理 |
| `created_at` | `TIMESTAMPTZ` | 否 | — | 当前 UTC | 创建时间 |
| `updated_at` | `TIMESTAMPTZ` | 否 | — | 当前 UTC | 每次更新自动刷新 |
| `completed_at` | `TIMESTAMPTZ` | 是 | — | — | 合并完成时间 |

### 4.2 索引与约束

| 名称 | 类型 | 定义 | 用途 |
|---|---|---|---|
| `thirdparty_upload_sessions_pkey` | 主键 | `(upload_id)` | 会话定位 |
| `thirdparty_upload_sessions_artifact_id_key` | 唯一 | `(artifact_id)` | 制品 ID 不重复（`NULL` 不参与） |
| `ix_thirdparty_upload_sessions_owner_user_id` | 索引 | `(owner_user_id)` | 按用户过滤 |
| `ix_thirdparty_upload_sessions_state` | 索引 | `(state)` | 清理任务按状态扫描 |
| `ix_thirdparty_upload_sessions_artifact_digest` | 索引 | `(artifact_digest)` | 按包摘要反查会话 |
| `ix_thirdparty_upload_sessions_expires_at` | 索引 | `(expires_at)` | 过期扫描 |
| `ck_upload_completed_has_artifact` | CHECK | `state NOT IN ('completed','consumed') OR (artifact_id IS NOT NULL AND artifact_digest IS NOT NULL AND artifact_path IS NOT NULL)` | 完成态必须有制品三件套 |
| `ck_upload_artifact_digest_length` | CHECK | `artifact_digest IS NULL OR length(artifact_digest) = 64` | 摘要必须是 64 位 SHA-256 |

### 4.3 状态字典

| 状态 | 含义 | 出口 |
|---|---|---|
| `uploading` | 分片接收中 | → `completing`（调用 complete）/ `canceled`（主动取消）/ `expired`（超时） |
| `completing` | 已加锁，正在合并分片并回填制品 | → `completed`；异常 → `failed` |
| `completed` | 合并完成，制品已落盘 | → `consumed`（被发布接口取走）；分片行此时可清理，会话行保留 |
| `consumed` | 已被发布流程消费，后续由软件包表接管 | 终态 |
| `failed` | 合并或校验失败 | 可继续传分片（回到 `uploading`）或再次 complete（复用已有分片） |
| `canceled` | 用户主动取消 | 终态，整行删除 |
| `expired` | 超过 `expires_at` 由清理任务置位 | 终态，整行删除 |

### 4.4 DDL

```sql
CREATE TABLE thirdparty_upload_sessions (
	upload_id         VARCHAR(64)  NOT NULL,
	owner_user_id     VARCHAR(64)  NOT NULL,
	uploaded_by       VARCHAR(255) NOT NULL,
	original_filename VARCHAR(512) NOT NULL,
	total_size        BIGINT       NOT NULL,
	chunk_size        INTEGER      NOT NULL,
	total_parts       INTEGER      NOT NULL,
	state             VARCHAR(32)  NOT NULL,
	artifact_id       VARCHAR(64),
	artifact_digest   VARCHAR(64),
	artifact_path     VARCHAR(1024),
	build_request_id  VARCHAR(64),
	last_error        TEXT,
	expires_at        TIMESTAMP WITH TIME ZONE NOT NULL,
	created_at        TIMESTAMP WITH TIME ZONE NOT NULL,
	updated_at        TIMESTAMP WITH TIME ZONE NOT NULL,
	completed_at      TIMESTAMP WITH TIME ZONE,
	PRIMARY KEY (upload_id),
	CONSTRAINT ck_upload_completed_has_artifact CHECK (
		state NOT IN ('completed', 'consumed') OR (
			artifact_id IS NOT NULL AND artifact_digest IS NOT NULL
			AND artifact_path IS NOT NULL
		)
	),
	CONSTRAINT ck_upload_artifact_digest_length CHECK (
		artifact_digest IS NULL OR length(artifact_digest) = 64
	),
	UNIQUE (artifact_id)
);

CREATE INDEX ix_thirdparty_upload_sessions_artifact_digest
	ON thirdparty_upload_sessions (artifact_digest);
CREATE INDEX ix_thirdparty_upload_sessions_expires_at
	ON thirdparty_upload_sessions (expires_at);
CREATE INDEX ix_thirdparty_upload_sessions_owner_user_id
	ON thirdparty_upload_sessions (owner_user_id);
CREATE INDEX ix_thirdparty_upload_sessions_state
	ON thirdparty_upload_sessions (state);
```

## 5. `thirdparty_upload_parts` — 上传分片

每个已接收且校验通过的分片一行。重复上传同一分片号是幂等的（命中已有行直接返回），因此该表同时充当断点续传的进度记录。

### 5.1 字段

| 字段 | 类型 | 可空 | 键/索引 | 应用层默认 | 说明 |
|---|---|---|---|---|---|
| `upload_id` | `VARCHAR(64)` | 否 | 主键，外键 | — | 所属会话，`ON DELETE CASCADE` |
| `part_number` | `INTEGER` | 否 | 主键 | — | 分片序号，从 1 开始 |
| `offset_bytes` | `BIGINT` | 否 | — | — | 该分片在整文件中的起始偏移 |
| `size_bytes` | `INTEGER` | 否 | — | — | 分片实际字节数，末片可小于 `chunk_size` |
| `content_digest` | `VARCHAR(64)` | 否 | — | — | 分片 SHA-256 十六进制串 |
| `created_at` | `TIMESTAMPTZ` | 否 | — | 当前 UTC | 接收时间 |

### 5.2 索引与约束

| 名称 | 类型 | 定义 | 用途 |
|---|---|---|---|
| `thirdparty_upload_parts_pkey` | 主键 | `(upload_id, part_number)` | 分片幂等去重、进度统计 |
| `thirdparty_upload_parts_upload_id_fkey` | 外键 | `upload_id` → `thirdparty_upload_sessions.upload_id` `ON DELETE CASCADE` | 会话删除时级联清理 |

主键前缀已覆盖 `upload_id` 查询，无需额外索引。**本表是临时数据**：合并完成后统一删除分片行，只保留会话行。

### 5.3 DDL

```sql
CREATE TABLE thirdparty_upload_parts (
	upload_id      VARCHAR(64) NOT NULL,
	part_number    INTEGER     NOT NULL,
	offset_bytes   BIGINT      NOT NULL,
	size_bytes     INTEGER     NOT NULL,
	content_digest VARCHAR(64) NOT NULL,
	created_at     TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (upload_id, part_number),
	FOREIGN KEY (upload_id) REFERENCES thirdparty_upload_sessions (upload_id)
		ON DELETE CASCADE
);
```

## 6. `local_agent_packages` — 本地软件包与上架状态

上架链路的核心表：一行代表一个**由内容摘要唯一标识的软件包**，承载其在管理面一侧的完整上架状态（构建、注册及其失败原因）。包身份即内容摘要，因此同一份文件重复上传会命中同一行：该包若尚未注册成功，发布接口会直接报冲突（提示当前状态与失败原因）；若已 `registered`，则不允许重复上架。

### 6.1 字段

| 字段 | 类型 | 可空 | 键/索引 | 应用层默认 | 说明 |
|---|---|---|---|---|---|
| `content_digest` | `VARCHAR(64)` | 否 | 主键 | — | 整包 SHA-256，包身份 |
| `package_path` | `VARCHAR(1024)` | 否 | 唯一 | — | 落盘路径 `{digest}.artifact`，与 `build_tasks.installer_path` 同值 |
| `original_filename` | `VARCHAR(512)` | 否 | — | — | 上传时的原始文件名 |
| `size_bytes` | `BIGINT` | 否 | — | — | 包字节数 |
| `uploaded_by` | `VARCHAR(255)` | 否 | — | — | 上传人用户名 |
| `access_mode` | `JSON` | 否 | — | — | 接入方式数组，结构见 §6.2 |
| `registration_name` | `VARCHAR(255)` | 否 | 复合唯一 | `''` | 管理员填写的 Agent 名称，注册中心 `name` 取值 |
| `card_version` | `VARCHAR(128)` | 是 | 复合唯一 | — | 卡片版本，来自镜像工厂解析结果；未解析时为 `NULL` |
| `registration_payload` | `JSON` | 是 | — | — | 发往注册中心的请求体快照，结构见 §6.2 |
| `description` | `TEXT` | 否 | — | `''` | 卡片描述，最长 1024 字符 |
| `state` | `VARCHAR(32)` | 否 | 索引 | `uploaded` | 上架状态，见 §6.4 |
| `last_error_message` | `TEXT` | 是 | — | — | 构建或注册失败原因 |
| `created_at` | `TIMESTAMPTZ` | 否 | — | 当前 UTC | 创建时间 |
| `updated_at` | `TIMESTAMPTZ` | 否 | — | 当前 UTC | 状态变更时间 |

### 6.2 JSON 列结构

`access_mode` 为对象数组，非空且 `name` 唯一；`port` 为 1–5 位字符串、取值 1–65535；`cmd` 长度 1–512。未显式提供端口时按名称取默认端口（`tui` → 2222，`web` → 4096）。

```json
[
  {"name": "tui", "port": "2222", "cmd": "<启动命令>"},
  {"name": "web", "port": "4096", "cmd": "<启动命令>"}
]
```

`registration_payload` 为发往注册中心的完整请求体快照，用于失败重试时原样复用：

| 键 | 说明 |
|---|---|
| `name` / `framework` | 同取 `registration_name`（注册中心契约要求二者一致） |
| `version` | 同 `card_version` |
| `description` / `uploaded_by` / `access_mode` / `package_path` | 上架上下文回填 |
| `env_vars` | 固定空对象 |
| `runtime_spec.rootfs.imageurl` | 构建产出镜像引用 |
| `image_module_version` | 镜像工厂返回的模块版本 |
| `image_archive_path` | 开启镜像归档时存在（`THIRDPARTY_AGENT_ARCHIVE_ENABLED`） |

### 6.3 索引与约束

| 名称 | 类型 | 定义 | 用途 |
|---|---|---|---|
| `local_agent_packages_pkey` | 主键 | `(content_digest)` | 包按内容去重 |
| `local_agent_packages_package_path_key` | 唯一 | `(package_path)` | 一个文件路径只对应一行 |
| `ix_local_agent_packages_state` | 索引 | `(state)` | 按状态筛选未注册包 |
| `uq_local_pkg_registry_identity_active` | **部分唯一** | `(registration_name, card_version) WHERE card_version IS NOT NULL AND state IN ('registering','registered')` | 注册中心 `name + version` 在「进行中/已注册」范围内独占；注册失败的包不占位，可换名字重试 |

### 6.4 状态字典

```text
uploaded ──> building ──> registering ──> registered
   ▲            │              │
   │            └──────────────┴──> build_failed / register_failed
   └────────────────────────────────────── 重新发布（复用包行）
```

| 状态 | 含义 | 出口 |
|---|---|---|
| `uploaded` | 已受理，等待构建；每次重新发布都会回到此态 | → `building` |
| `building` | 镜像工厂构建中 | → `registering`；失败 → `build_failed` |
| `registering` | 已向注册中心提交注册（含超时待确认） | → `registered`；失败 → `register_failed` |
| `registered` | 注册成功，卡片可见 | 终态，不可重复上架 |
| `build_failed` | 构建失败，`last_error_message` 有原因 | 可重新发布 |
| `register_failed` | 注册失败（含身份冲突） | 可重新发布；不占用名称/版本 |

进程重启时的恢复动作：`registering` 的包会被回收（查注册中心确认是否已注册成功，据此收敛到 `registered` 或 `register_failed`）；仍处 `pending`/`building` 的构建任务统一置为 `failed`，且其对应包若停在 `uploaded`/`building`，一并转为 `build_failed`（错误信息为后端重启导致）。

### 6.5 DDL

```sql
CREATE TABLE local_agent_packages (
	content_digest       VARCHAR(64)  NOT NULL,
	package_path         VARCHAR(1024) NOT NULL,
	original_filename    VARCHAR(512) NOT NULL,
	size_bytes           BIGINT       NOT NULL,
	uploaded_by          VARCHAR(255) NOT NULL,
	access_mode          JSON         NOT NULL,
	registration_name    VARCHAR(255) NOT NULL,
	card_version         VARCHAR(128),
	registration_payload JSON,
	description          TEXT         NOT NULL,
	state                VARCHAR(32)  NOT NULL,
	last_error_message   TEXT,
	created_at           TIMESTAMP WITH TIME ZONE NOT NULL,
	updated_at           TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (content_digest),
	UNIQUE (package_path)
);

CREATE INDEX ix_local_agent_packages_state
	ON local_agent_packages (state);

CREATE UNIQUE INDEX uq_local_pkg_registry_identity_active
	ON local_agent_packages (registration_name, card_version)
	WHERE card_version IS NOT NULL
	  AND state IN ('registering', 'registered');
```

## 7. `build_tasks` — 镜像构建任务

一次构建尝试一行。同一软件包可有多行（失败后重试会新建任务行），表因此同时承担**任务状态、并发锁、构建历史**三种职责。

### 7.1 字段

| 字段 | 类型 | 可空 | 键/索引 | 应用层默认 | 说明 |
|---|---|---|---|---|---|
| `task_id` | `VARCHAR(64)` | 否 | 主键 | — | 构建请求 ID，格式 `build-{12位十六进制}` |
| `installer_path` | `VARCHAR(1024)` | 否 | 部分唯一 | — | 输入软件包路径，同 `local_agent_packages.package_path` |
| `status` | `VARCHAR(32)` | 否 | 部分唯一 | `pending` | 任务状态，见 §7.3 |
| `progress` | `INTEGER` | 否 | — | `0` | 构建进度百分比 0–100 |
| `image` | `VARCHAR(512)` | 是 | — | — | 构建产出的镜像引用 |
| `image_digest` | `VARCHAR(128)` | 是 | — | — | 镜像摘要 |
| `created_at` | `TIMESTAMPTZ` | 否 | — | 当前 UTC | 任务创建时间 |
| `started_at` | `TIMESTAMPTZ` | 是 | — | — | 开始构建时间 |
| `finished_at` | `TIMESTAMPTZ` | 是 | — | — | 结束时间（成功或失败） |
| `error_message` | `VARCHAR(1024)` | 是 | — | — | 失败原因，写入时截断至 1024 |

### 7.2 索引与约束

| 名称 | 类型 | 定义 | 用途 |
|---|---|---|---|
| `build_tasks_pkey` | 主键 | `(task_id)` | 任务定位 |
| `uq_build_task_active_path` | **部分唯一** | `(installer_path) WHERE status IN ('pending','building')` | 同一软件包同时只允许一个未完成构建，是并发的数据库级兜底 |

并发还受应用层约束：全局同时构建数上限为 2（超出抛 `ConcurrentBuildLimitError`）。

### 7.3 状态字典

| 状态 | 含义 | 出口 |
|---|---|---|
| `pending` | 已创建，尚未开始 | → `building` |
| `building` | 构建中，`progress` 由回调更新 | → `done`；失败 → `failed` |
| `done` | 构建成功，`image` / `image_digest` 已写入 | 终态 |
| `failed` | 构建失败、被新任务取代，或进程重启后的中断回收 | 终态 |

### 7.4 DDL

```sql
CREATE TABLE build_tasks (
	task_id        VARCHAR(64)   NOT NULL,
	installer_path VARCHAR(1024) NOT NULL,
	status         VARCHAR(32)   NOT NULL,
	progress       INTEGER       NOT NULL,
	image          VARCHAR(512),
	image_digest   VARCHAR(128),
	created_at     TIMESTAMP WITH TIME ZONE NOT NULL,
	started_at     TIMESTAMP WITH TIME ZONE,
	finished_at    TIMESTAMP WITH TIME ZONE,
	error_message  VARCHAR(1024),
	PRIMARY KEY (task_id)
);

CREATE UNIQUE INDEX uq_build_task_active_path
	ON build_tasks (installer_path)
	WHERE status IN ('pending', 'building');
```

## 8. 建表与演进方式

**不使用 Alembic 迁移**，全部在应用启动时幂等执行。启动序列见 `app/main.py`：

```text
init_engine()                      # 创建共享异步引擎
└─ _create_log_tables(engine)      # Base.metadata.create_all（覆盖全部已注册模型）
└─ ensure_thirdparty_agent_tables(engine)   # 上架 4 张表的权威入口
   ├─ create_all(tables=[4 张表])
   ├─ ALTER TABLE local_agent_packages ADD COLUMN IF NOT EXISTS ...   # 列级补丁
   ├─ 收敛历史重复活跃构建任务（ROW_NUMBER 分区后置为 failed）
   ├─ CREATE UNIQUE INDEX IF NOT EXISTS uq_build_task_active_path
   ├─ CREATE UNIQUE INDEX IF NOT EXISTS uq_local_pkg_registry_identity_active
   └─ DROP INDEX IF EXISTS uq_local_pkg_registry_identity            # 旧索引
```

列级补丁清单（`app/thirdparty_agent/engine.py`）：

| 语句 | 目标 |
|---|---|
| `ADD COLUMN IF NOT EXISTS description TEXT DEFAULT ''` | 卡片描述 |
| `ADD COLUMN IF NOT EXISTS registration_name VARCHAR(255) NOT NULL DEFAULT ''` | 注册身份 |
| `ADD COLUMN IF NOT EXISTS card_version VARCHAR(128)` | 卡片版本 |
| `ADD COLUMN IF NOT EXISTS registration_payload JSON` | 注册请求快照 |
| `DROP INDEX IF EXISTS uq_local_pkg_registry_identity` | 旧身份唯一索引语义过宽（含 `register_failed`），由新索引取代 |

两点需要留意：

1. **默认值分两层。** 状态、时间戳、进度等默认值都由应用层写入，DDL 中没有 `DEFAULT` 子句；只有 `description` 与 `registration_name` 在两个 `ADD COLUMN` 语句里带了 `DEFAULT ''`，因此**升级库**上这两列有数据库默认值，**全新建库**（`create_all`）则没有。直接写库的脚本不应依赖任一默认值。
2. **唯一约束是部分索引。** `uq_local_pkg_registry_identity_active` 与 `uq_build_task_active_path` 都带 `WHERE` 条件，只在活跃状态内生效；历史行（如已 `registered` 的旧版本、已 `failed` 的旧任务）不参与冲突判定。

## 9. 生命周期与清理

| 数据 | 保留策略 | 清理动作 |
|---|---|---|
| `thirdparty_upload_parts` | 会话结束后即清理 | 合并完成、取消、过期时按 `upload_id` 删除 |
| `thirdparty_upload_sessions`（未完成） | TTL 3600 秒，清理周期 600 秒 | 置 `expired` 后删除会话行与分片行 |
| `thirdparty_upload_sessions`（`completed`/`consumed`） | 长期保留 | 保留会话行（持有权威整文件摘要），仅删分片；分片清理失败还会重试 |
| `thirdparty_upload_sessions`（`canceled`） | 立即 | 删除会话行与分片行 |
| `local_agent_packages` | 无 TTL，随卡片删除 | 删除卡片时一并清理包行、构建历史与磁盘文件（含 sidecar） |
| `build_tasks` | 无 TTL，保留完整构建历史 | 随软件包删除 |

过期清理使用 `SELECT ... FOR UPDATE SKIP LOCKED` 抢占待清理会话，多实例并发安全。

## 10. 排障速查

```sql
-- 当前未注册完成的包（卡片缺失排查入口）
SELECT content_digest, registration_name, card_version, state, last_error_message
FROM local_agent_packages
WHERE state <> 'registered'
ORDER BY updated_at DESC;

-- 某软件包的完整构建历史
SELECT task_id, status, progress, image, error_message, created_at, finished_at
FROM build_tasks
WHERE installer_path LIKE '%<digest>.artifact'
ORDER BY created_at DESC;

-- 卡在活跃态、阻塞重试的记录（部分唯一索引冲突来源）
SELECT 'session' AS kind, upload_id AS id, state FROM thirdparty_upload_sessions
WHERE state IN ('uploading', 'completing')
UNION ALL
SELECT 'build', task_id, status FROM build_tasks
WHERE status IN ('pending', 'building');
```

## 11. 不落库的上架状态

| 状态 | 位置 | 说明 |
|---|---|---|
| 启动模板、表单格式、模板修订号 | `{package_path 同目录}/{digest}.metadata.json` | sidecar 结构版本 3；模板为可选项，未配置时按 Agent 自身默认值启动 |
| 用户级模型 API Key | 用户家目录下由模板 `relative_path` 指向的 `.env` | 环境变量名固定为 `MODEL_API_KEY`，不写入数据库与模板正文 |
| 卡片、实例、默认版本、运行规格 | Agent 注册中心 | 管理面数据库不保存卡片投影 |

## 12. 参考代码位置

| 内容 | 位置 |
|---|---|
| 4 张表的 ORM 定义 | `backend/app/models/thirdparty_agent.py` |
| 表注册（导入即注册进 metadata） | `backend/app/models/__init__.py` |
| 建表与列级补丁 | `backend/app/thirdparty_agent/engine.py` |
| 启动序列调用点 | `backend/app/main.py`（`ensure_thirdparty_agent_tables`） |
| 上传会话与分片逻辑、过期清理 | `backend/app/services/chunked_upload_service.py` |
| 上架状态机、构建与注册编排 | `backend/app/services/thirdparty_agent_service.py` |
| 启动模板与模型 API Key 文件读写 | `backend/app/thirdparty_agent/launch_config.py` |
| 落盘门禁与 sidecar 读写 | `backend/app/thirdparty_agent/upload_gate.py` |
| 上传上限、TTL、清理周期等参数 | `backend/app/config.py`（`THIRDPARTY_AGENT_*`） |
| 数据边界设计依据 | [三方 Agent 新旧数据库对比与变更计划](third-party-agent-database-comparison-and-migration-plan.md) |
