---
title: "三方 Agent 上架服务数据库表结构（管理面现状）"
type: reference
domain: agent
status: active
---

# 三方 Agent 上架服务数据库表结构（管理面现状）

> 现状记录，随代码更新；覆盖管理面「三方 Agent 上架」链路（上传 → 构建 → 注册）。  
> 代码基线：`AgentBox-Manager`，`feature/thirdparty-access-mode-image`，`693a0fe`  
> 数据库：PostgreSQL 18（`AGENTOS_DATABASE_URL`，与 LiteLLM 共享 `agentos` 库）；表定义：`backend/app/models/thirdparty_agent.py`  
> 关联文档：[新旧数据库对比与变更计划](third-party-agent-database-comparison-and-migration-plan.md)、[第三方 Agent 接入架构](third-party-agent-access-architecture.md)

管理面只保存**未注册成功的包与构建历史**，卡片与实例的权威数据在注册中心（管理面经 `image_process_client` 读写）。

## 1. 表清单

| 表名 | ORM 类 | 用途 | 主键 | 状态列 |
|---|---|---|---|---|
| `thirdparty_upload_sessions` | `UploadSession` | 分片上传会话，一行一次上传 | `upload_id` | `state` |
| `thirdparty_upload_parts` | `UploadPart` | 上传分片，一行一个分片 | `upload_id` + `part_number` | — |
| `local_agent_packages` | `LocalAgentPackage` | 本地包与上架状态，一行一个包 | `content_digest` | `state` |
| `build_tasks` | `BuildTask` | 镜像构建任务，一行一次尝试 | `task_id` | `status` |

字段表中的缩写：`PK` 主键、`UK` 唯一、`IDX` 普通索引、`UK(部分)` 带 `WHERE` 条件的部分唯一索引；类型列带 `NOT NULL` 表示非空，未标注默认可空；默认值由应用层写入（见 §7）。正文只写列级索引信息，PostgreSQL 自动生成的对象名按惯例即可推出（主键 `{表名}_pkey`、唯一 `{表名}_{列名}_key`、外键 `{表名}_{列名}_fkey`、普通索引 `ix_{表名}_{列名}`）。

## 2. 表关系

```text
upload_sessions ──1:N──> upload_parts       唯一物理外键，ON DELETE CASCADE
       │ artifact_digest = content_digest
       ▼
local_agent_packages ──1:N──> build_tasks   package_path = installer_path
```

除分片 → 会话这一处外键外，其余关联都是**值对应关系**（会话到包按摘要、包到构建按路径、会话到构建按 `build_request_id`）。因此**不能用级联删除管理生命周期**：删除包行需自行清理对应的 `build_tasks` 行与磁盘文件。

## 3. `thirdparty_upload_sessions` — 分片上传会话

分片全部校验通过并合并后，在**同一行**回填制品三件套（`artifact_id` / `artifact_digest` / `artifact_path`），供发布接口按 artifact 取包。

| 字段 | 类型 | 键 | 说明 |
|---|---|---|---|
| `upload_id` | `VARCHAR(64) NOT NULL` | PK | 上传会话 ID，服务端生成 |
| `owner_user_id` | `VARCHAR(64) NOT NULL` | IDX | 归属用户 ID，读取时校验以隔离多用户 |
| `uploaded_by` | `VARCHAR(255) NOT NULL` | | 上传人用户名 |
| `original_filename` | `VARCHAR(512) NOT NULL` | | 客户端原始文件名，仅展示 |
| `total_size` | `BIGINT NOT NULL` | | 声明总字节数，应用层限制 ≤ 2 GiB |
| `chunk_size` | `INTEGER NOT NULL` | | 分片大小，来自 `THIRDPARTY_AGENT_UPLOAD_CHUNK_BYTES`（64 MiB） |
| `total_parts` | `INTEGER NOT NULL` | | 分片总数，由 `total_size` 与分片大小推导 |
| `state` | `VARCHAR(32) NOT NULL` | IDX | 默认 `uploading`，取值见下 |
| `artifact_id` | `VARCHAR(64)` | UK | 制品 ID，合并完成后回填 |
| `artifact_digest` | `VARCHAR(64)` | IDX | 整文件 SHA-256 十六进制串，与包主键同值 |
| `artifact_path` | `VARCHAR(1024)` | | 包落盘绝对路径 `.../packages/{digest}.artifact` |
| `build_request_id` | `VARCHAR(64)` | | 发布受理后写入，对应 `build_tasks.task_id` |
| `last_error` | `TEXT` | | 最近一次失败原因 |
| `expires_at` | `TIMESTAMP WITH TIME ZONE NOT NULL` | IDX | 当前时刻 + TTL（3600 秒）；进入 `completing` 时刷新，避免合并期间被清理 |
| `created_at` | `TIMESTAMP WITH TIME ZONE NOT NULL` | | 创建时间 |
| `updated_at` | `TIMESTAMP WITH TIME ZONE NOT NULL` | | 每次更新自动刷新 |
| `completed_at` | `TIMESTAMP WITH TIME ZONE` | | 合并完成时间 |

**索引与约束**：PK `upload_id`；UK `artifact_id`（`NULL` 不参与）；索引 `owner_user_id` / `state` / `artifact_digest` / `expires_at`；CHECK `ck_upload_completed_has_artifact`（`state` 为 `completed`/`consumed` 时制品三件套必须非空）、`ck_upload_artifact_digest_length`（`artifact_digest` 非空时必须为 64 字符）。

**状态**：`uploading → completing → completed → consumed`；失败分支 `failed`（可续传或重新合并）、`canceled`、`expired`。仅 `uploading` 与 `failed` 可继续传分片、发起合并（代码中的 `_ACTIVE_STATES`），`completing` 已加锁，重复合并返回冲突。

## 4. `thirdparty_upload_parts` — 上传分片

重复上传同一分片号是幂等的（命中已有行直接返回），该表因此同时是断点续传的进度记录。

| 字段 | 类型 | 键 | 说明 |
|---|---|---|---|
| `upload_id` | `VARCHAR(64) NOT NULL` | PK, FK | 所属会话，`ON DELETE CASCADE` |
| `part_number` | `INTEGER NOT NULL` | PK | 分片序号，从 1 开始 |
| `offset_bytes` | `BIGINT NOT NULL` | | 该分片在整文件中的起始偏移 |
| `size_bytes` | `INTEGER NOT NULL` | | 分片实际字节数，末片可小于 `chunk_size` |
| `content_digest` | `VARCHAR(64) NOT NULL` | | 分片 SHA-256 十六进制串 |
| `created_at` | `TIMESTAMP WITH TIME ZONE NOT NULL` | | 接收时间 |

**索引与约束**：PK `(upload_id, part_number)`（前缀已覆盖按会话查询，无需额外索引）；FK `upload_id` → `thirdparty_upload_sessions.upload_id` `ON DELETE CASCADE`。**本表是临时数据**：合并完成后统一删除分片行，只保留会话行。

## 5. `local_agent_packages` — 本地软件包与上架状态

上架链路的核心表，一行代表一个**由内容摘要唯一标识的软件包**，承载它在管理面一侧的完整上架状态（构建、注册及失败原因）。包身份即内容摘要，因此同一份文件重复上传会命中同一行：若尚未注册成功，发布接口直接报冲突；若已 `registered`，不允许重复上架。

| 字段 | 类型 | 键 | 说明 |
|---|---|---|---|
| `content_digest` | `VARCHAR(64) NOT NULL` | PK | 整包 SHA-256，包身份 |
| `package_path` | `VARCHAR(1024) NOT NULL` | UK | 落盘路径 `{digest}.artifact`，与 `build_tasks.installer_path` 同值 |
| `original_filename` | `VARCHAR(512) NOT NULL` | | 上传时的原始文件名 |
| `size_bytes` | `BIGINT NOT NULL` | | 包字节数 |
| `uploaded_by` | `VARCHAR(255) NOT NULL` | | 上传人用户名 |
| `access_mode` | `JSON NOT NULL` | | 接入方式数组，见下 |
| `registration_name` | `VARCHAR(255) NOT NULL` | UK(部分) | 管理员填写的 Agent 名称，注册中心 `name` 取值；默认 `''` |
| `card_version` | `VARCHAR(128)` | UK(部分) | 卡片版本，来自镜像工厂解析结果；未解析时为 `NULL` |
| `registration_payload` | `JSON` | | 发往注册中心的请求体快照，见下 |
| `description` | `TEXT NOT NULL` | | 卡片描述，默认 `''`，最长 1024 字符 |
| `state` | `VARCHAR(32) NOT NULL` | IDX | 默认 `uploaded`，取值见下 |
| `last_error_message` | `TEXT` | | 构建或注册失败原因 |
| `created_at` | `TIMESTAMP WITH TIME ZONE NOT NULL` | | 创建时间 |
| `updated_at` | `TIMESTAMP WITH TIME ZONE NOT NULL` | | 状态变更时间 |

**索引与约束**：PK `content_digest`（按内容去重）；UK `package_path`；索引 `state`；部分唯一索引 `uq_local_pkg_registry_identity_active` = `(registration_name, card_version)` `WHERE card_version IS NOT NULL AND state IN ('registering','registered')`，保证注册中心的 `name + version` 在「进行中/已注册」范围内独占，注册失败的包不占位、可改名重试。

**状态**：`uploaded → building → registering → registered`；失败落到 `build_failed` / `register_failed`，可重新发布（回到 `uploaded` 复用包行）；`registered` 为终态。进程重启时 `registering` 的包会回查注册中心收敛到 `registered` 或 `register_failed`，活跃构建任务置 `failed` 且对应包置 `build_failed`。

**JSON 列结构**：

```json
[{"name": "tui", "port": "2222", "cmd": "<启动命令>"},
 {"name": "web", "port": "4096", "cmd": "<启动命令>"}]
```

`access_mode` 为对象数组，`name` 非空且唯一，`port` 为 1–5 位字符串（取值 1–65535），缺省时按名称取默认端口（`tui` → 2222，`web` → 4096），`cmd` 长度 1–512。`registration_payload` 是注册请求体快照，供失败重试原样复用，键包括：`name` / `framework`（同 `registration_name`）、`version`（同 `card_version`）、`description` / `uploaded_by` / `access_mode` / `package_path`（上架上下文回填）、`env_vars`（固定空对象）、`runtime_spec.rootfs.imageurl`（构建产出镜像）、`image_module_version`，开启镜像归档（`THIRDPARTY_AGENT_ARCHIVE_ENABLED`）时另有 `image_archive_path`。

## 6. `build_tasks` — 镜像构建任务

一次构建尝试一行，同一软件包可有多行（失败重试会新建任务行），该表同时承担**任务状态、并发锁、构建历史**三种职责。

| 字段 | 类型 | 键 | 说明 |
|---|---|---|---|
| `task_id` | `VARCHAR(64) NOT NULL` | PK | 构建请求 ID，格式 `build-{12位十六进制}` |
| `installer_path` | `VARCHAR(1024) NOT NULL` | UK(部分) | 输入软件包路径，同 `local_agent_packages.package_path` |
| `status` | `VARCHAR(32) NOT NULL` | UK(部分) | 默认 `pending`，取值见下 |
| `progress` | `INTEGER NOT NULL` | | 构建进度百分比 0–100，默认 `0` |
| `image` | `VARCHAR(512)` | | 构建产出的镜像引用 |
| `image_digest` | `VARCHAR(128)` | | 镜像摘要 |
| `created_at` | `TIMESTAMP WITH TIME ZONE NOT NULL` | | 任务创建时间 |
| `started_at` | `TIMESTAMP WITH TIME ZONE` | | 开始构建时间 |
| `finished_at` | `TIMESTAMP WITH TIME ZONE` | | 结束时间（成功或失败） |
| `error_message` | `VARCHAR(1024)` | | 失败原因，写入时截断至 1024 |

**索引与约束**：PK `task_id`；部分唯一索引 `uq_build_task_active_path` = `(installer_path)` `WHERE status IN ('pending','building')`，同一软件包同时只允许一个未完成构建，是并发的数据库级兜底（应用层另有全局并发上限 2）。

**状态**：`pending → building → done`；失败或被新任务取代 → `failed`。

## 7. 建表方式

无 Alembic 迁移，全部在应用启动时幂等执行：`ensure_thirdparty_agent_tables()` 做 `create_all`，再用 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 补列（`description`、`registration_name`、`card_version`、`registration_payload`），同时创建上述两个部分唯一索引并废弃旧索引 `uq_local_pkg_registry_identity`。

两点需要留意：

1. 默认值都在应用层写入，DDL 里没有 `DEFAULT` 子句；只有 `description` 与 `registration_name` 因 `ADD COLUMN` 语句携带 `DEFAULT ''`，**升级库**有数据库默认值而**全新建库**没有，写库脚本不要依赖默认值。
2. 两个唯一约束都是**部分索引**（带 `WHERE`），只在活跃状态内生效，历史行（已注册的旧版本、已失败的任务）不参与冲突判定。

## 8. 生命周期与清理

| 数据 | 保留策略 |
|---|---|
| `thirdparty_upload_parts` | 会话结束时删行；合并完成、取消、过期时触发 |
| `thirdparty_upload_sessions`（未完成） | TTL 3600 秒、清理周期 600 秒，置 `expired` 后删会话行与分片行 |
| `thirdparty_upload_sessions`（`completed`/`consumed`） | 保留会话行（持有权威整文件摘要），仅删分片；分片清理失败会重试 |
| `thirdparty_upload_sessions`（`canceled`） | 立即删会话行与分片行 |
| `local_agent_packages` / `build_tasks` | 无 TTL，随卡片删除一并清理，含磁盘文件 |

过期清理使用 `SELECT ... FOR UPDATE SKIP LOCKED` 抢占待清理会话，多实例并发安全。

## 9. 参考位置

- 表定义：`backend/app/models/thirdparty_agent.py`；表注册：`backend/app/models/__init__.py`
- 建表与补列：`backend/app/thirdparty_agent/engine.py`；启动调用点：`backend/app/main.py`
- 会话与分片、过期清理：`backend/app/services/chunked_upload_service.py`
- 上架状态机、构建与注册编排：`backend/app/services/thirdparty_agent_service.py`
- 包落盘门禁与摘要校验：`backend/app/thirdparty_agent/upload_gate.py`
- 参数（上传上限、TTL、清理周期）：`backend/app/config.py` 的 `THIRDPARTY_AGENT_*`
