---
title: "三方智能体文件上传现状分析与演进计划"
type: design
domain: agent
status: draft
date: 2026-09-03
---

# 三方智能体文件上传现状分析与演进计划

## 1. 文档目的

本文基于 AgentBox-Manager 当前 `refactor/file_upload` 分支，对三方智能体文件上传、镜像构建、注册、查询、重试和清理链路进行完整分析，并与最新数据库迁移设计进行对照。

本文回答以下问题：

1. 当前上传后的实际处理流程是什么；
2. 当前同步和异步异常如何处理；
3. 当前实现存在哪些确定问题和风险；
4. 新数据库设计可以解决哪些问题；
5. 哪些问题不能仅靠更换数据库解决；
6. 后续应该如何分阶段实施、迁移、测试和验收。

本文不直接修改实现。问题等级用于安排开发优先级，不表示所有问题均已在线上复现。

## 2. 分析范围与基线

### 2.1 代码基线

- 仓库：`AgentBox-Manager`
- 分支：`refactor/file_upload`
- 提交：`599db0e4237245d91118c60f0db790d42f66f976`
- 分析日期：2026-09-03

### 2.2 主要代码范围

管理面后端：

- `backend/app/api/v1/thirdparty_agent.py`
- `backend/app/services/thirdparty_agent_service.py`
- `backend/app/thirdparty_agent/upload_gate.py`
- `backend/app/thirdparty_agent/local_file_cleaner.py`
- `backend/app/models/thirdparty_agent.py`
- `backend/app/services/image_process_client.py`
- `backend/app/services/agent_register_client.py`

镜像工厂：

- `image_process/app/tasks.py`
- `image_process/app/factory/*`
- `image_process/app/factory/recipes/npm_tgz.py`
- `image_process/app/builder.py`

前端：

- `frontend/src/views/resources/agent/FrameworkPage.vue`
- `frontend/src/api/framework.ts`

### 2.3 参考设计

本文主要对照：

- [`third-party-agent-database-comparison-and-migration-plan.md`](./third-party-agent-database-comparison-and-migration-plan.md)
- [`third-party-agent-artifact-image-factory-design-v2.md`](../image-factory/third-party-agent-artifact-image-factory-design-v2.md)
- [`third-party-agent-integration-guide.md`](./third-party-agent-integration-guide.md)

其中数据库迁移文档状态为 `draft`。正式开发前仍需冻结新版注册中心 OpenAPI、字段含义和幂等语义。

### 2.4 验证限制

本次完成了代码和测试用例静态审查。当前执行环境没有可用的 `uv` 或 `python` 命令，未能实际运行 pytest。因此，本文会区分代码可直接证明的问题、并发/故障窗口风险和设计层问题。

## 3. 当前架构与数据职责

当前链路涉及四类状态载体：

```text
浏览器
  └─ 保存当前上传弹窗、轮询进度等临时 UI 状态

管理面数据库
  ├─ build_tasks：任务状态、进度、并发锁、构建历史、未注册包索引
  └─ agent_registrations：注册中心卡片缺失字段的本地补充记录

image-process 内存
  └─ _tasks：镜像构建实时状态，进程重启即丢失

注册中心
  └─ image/card：已注册卡片、运行规格和默认版本
```

当前最主要的结构性问题是：

- `build_tasks` 同时承担四种不同职责；
- “未注册包”不是显式实体，而是通过表间路径差集推导；
- 已注册卡片同时依赖注册中心与本地 `agent_registrations`；
- 任务实时状态分散在管理面数据库、管理面协程和 image-process 内存中；
- 跨数据库、文件系统、Docker 和注册中心的操作没有明确的阶段状态与补偿模型。

## 4. 当前上传及后续处理逻辑

### 4.1 前端提交

管理员在三方智能体管理页选择文件并填写 `launch_command`。前端进行以下校验：

- 必须选择文件；
- 启动命令去除首尾空白后不能为空；
- 文件大小不得超过前端写死的 500 MB。

前端通过 multipart form-data 调用：

```text
POST /api/v1/thirdparty_agent/cards
  package=<File>
  launch_command=<string>
```

请求超时设置为一小时，但后端通常在文件落盘和任务入库后立即返回 202，而不是等待构建完成。

### 4.2 API 接入和权限

上传接口要求管理员权限。FastAPI 对 `launch_command` 声明长度为 1～512，服务层再次执行 `strip()` 并拒绝纯空白字符串。

领域异常映射为 HTTP 状态：

| 异常 | HTTP | 当前语义 |
|---|---:|---|
| `InvalidUploadError` | 400 | 文件名、空文件或启动命令非法 |
| `PackageTooLargeError` | 400 | 文件超过后端限制 |
| `PackageLockedError` | 409 | 同包正在构建或全局并发已满 |
| `CardHasInstancesError` | 409 | 卡片仍有关联实例 |
| `DefaultVersionProtectedError` | 409 | 默认版本保护 |
| `AgentNotFoundError` | 404 | 卡片、包或任务不存在 |
| `InsufficientDiskSpaceError` | 507 | 本地磁盘空间不足 |
| `AgentServiceError` | 502 | 注册中心服务不可用 |
| 其他三方智能体领域异常 | 500 | 未明确分类的服务错误 |

### 4.3 通用上传门禁与落盘

`UploadGate.persist()` 当前执行：

1. 检查原始文件名是否包含 `/`、`\`，或等于 `.`、`..`；
2. 如果 multipart 提供了可信的声明大小，先做一次大小判断；
3. 使用 `await uploaded.read()` 将整个文件读入 Python 内存；
4. 拒绝空文件；
5. 根据实际内存字节数再次校验最大 500 MB；
6. 检查目标磁盘剩余空间是否大于文件大小加 50 MB；
7. 计算内容 SHA-256；
8. 将文件保存为 `<sha256>.artifact`；
9. 将原始文件名写入 `<sha256>.metadata.json`；
10. 临时文件写完后通过 `os.replace()` 分别替换制品和 metadata。

文件格式、包结构、平台、架构和 libc 不在上传门禁校验，而是在 image-process Recipe 中校验。这一职责划分符合 Recipe 扩展设计。

### 4.4 构建任务准入

文件成功落盘后，管理面生成 `build-<12位随机标识>`，创建 `BuildTask`：

```text
status = pending
progress = 0
installer_path = <digest>.artifact
```

`BuildTask.try_insert()` 依次：

1. 查询同一 `installer_path` 是否存在 `pending/building`；
2. 统计全库 `pending/building` 数量；
3. 当前默认最多允许两个活动任务；
4. 插入并提交任务。

准入成功后，管理面用 `asyncio.create_task()` 启动本进程后台协程，并向前端返回：

```json
{
  "digest": "<sha256>",
  "request_id": "build-..."
}
```

### 4.5 管理面与 image-process 的构建交互

管理面后台协程：

1. 将本地任务标记为 `building`；
2. 调用 image-process `POST /v1/builds`，传入共享文件路径和 request ID；
3. 每秒调用 `GET /v1/builds/{request_id}`；
4. 将远端进度同步到本地 `build_tasks.progress`；
5. 远端状态为 `done/failed` 时退出轮询。

image-process 对任务的管理完全在内存 `_tasks` 中。接单后也通过 `asyncio.create_task()` 运行构建，完成任务默认保留 24 小时，只有触发新任务时才执行过期清理。

### 4.6 Recipe 识别和镜像构建

当前默认仅注册 `npm_tgz_on_base` Recipe。主要过程为：

1. 尝试以 gzip tar 打开制品；
2. 在归档中寻找路径以 `package/package.json` 结尾的成员；
3. 读取 `name/version`；
4. 从 npm 包名后缀解析 OS、架构和 libc；
5. 与构建宿主机平台严格匹配；
6. 规范化 agent name 并校验 Docker tag 字符；
7. 选择配置的 `agent-base`；
8. 创建 request ID 对应的临时工作目录；
9. 拷贝上传制品和 `agent.Dockerfile`；
10. 执行 `docker build`；
11. 如果启用归档，执行 `docker save | gzip`；
12. inspect 最终镜像，形成 `BuildResult`；
13. finally 删除临时工作目录。

`BuildResult` 包含名称、版本、镜像 tag、归档路径、runtime spec、Recipe、基础镜像和镜像摘要等信息。

### 4.7 注册中心登记与本地补充记录

镜像构建成功后，管理面组装注册请求。由于当前注册中心缺少独立 `launch_command` 字段，当前实现临时把启动命令写入 `framework`：

```text
framework = launch_command
framework_version = 制品 version
runtime_spec.rootfs.imageurl = 构建镜像 tag
```

调用顺序为：

1. 注册中心创建或更新镜像记录；
2. 本地 `AgentRegistration.upsert()` 写入 package path、agent name 等补充数据；
3. 本地 `BuildTask` 标记为 `done`，进度设为 100。

当前注册中心不保存或不返回全部本地字段，所以卡片查询需要把注册中心结果与 `agent_registrations` 合并。

### 4.8 前端状态轮询

前端拿到 digest 后，每两秒调用：

```text
GET /api/v1/thirdparty_agent/unregistered/{digest}
```

当前完成判断为：

- `last_error` 非空且 `locked=false`：展示失败；
- `last_error` 为空且 `locked=false`：认为上架成功；
- 请求异常：停止轮询、关闭构建窗口并刷新列表。

前端实际没有用首次响应中的 request ID 查询，而是按 digest 查询该路径下最新任务。

### 4.9 失败包重试与删除

“未注册包”列表并非独立表，而是：

1. 查询全部 `AgentRegistration.installer_path`；
2. 查询全部 `BuildTask`，每个路径只保留最新任务；
3. 排除已经出现在注册记录中的路径；
4. 排除本地文件已不存在的路径。

重试时要求管理员重新填写启动命令，创建新的 `BuildTask` 并再次完整构建。

删除失败包时，只要同路径没有 `pending/building`，就删除 artifact、metadata 和该路径的所有构建任务。

## 5. 当前问题清单

### 5.1 P0：进程重启导致任务永久锁定

#### 现象

管理面任务入库后通过进程内 `asyncio.create_task()` 推进。若管理面进程退出，协程丢失，但数据库状态仍为 `pending/building`。

image-process 同样使用内存任务。重启后管理面再次查询该 request ID，将得到 404；客户端把 404 转为 `None`。

#### 影响

- 包无法重试；
- 包无法删除；
- 僵尸任务永久占用全局并发名额；
- 两个僵尸任务即可阻止所有新上传；
- 前端可能永久显示构建中。

#### 根因

- 没有任务心跳或 `updated_at`；
- 没有锁租约；
- 没有启动恢复；
- 没有构建总超时；
- 活动状态同时被当作永久互斥锁。

### 5.2 P0：远端状态查询失败会无限轮询

`ImageProcessClient.fetch_build()` 对网络异常、404 和所有 HTTP 错误统一返回 `None`。管理面收到 `None` 后无条件每秒继续查询，没有最大次数或截止时间。

短暂错误与永久丢失无法区分，导致任务永不进入 `failed`，并继续占用锁和并发名额。

### 5.3 P1：注册中心成功后本地状态可能暂时不一致

注册中心成功之后，仍可能在以下步骤失败：

- 本地 `AgentRegistration.upsert()`；
- 本地 `BuildTask.mark_done()`；
- 数据库连接或提交。

此时注册中心已经存在卡片，本地却显示失败或缺少补充信息。当前总异常处理只会把任务标为失败，没有：

- 按 `name + version` 回读确认；
- 对同主键重复提交的内容等价与冲突语义；
- 注册成功状态快照；
- 后台对账。

新版注册中心已经明确以 `name + version` 为联合主键，因此注册阶段并不缺少业务幂等键，也不需要为了当前场景额外引入 `operation_id`。只要注册中心约定“同主键且内容等价时返回已有结果，同主键但关键内容不一致时返回 409”，重复注册不会产生第二张卡片。这里的主要风险是管理面没有在超时或本地落库失败后执行回读确认，可能重复构建或把实际成功误报为失败。基于现有业务约束，该问题风险由 P0 下调为 P1。

### 5.4 P1：大文件整包读入内存并同步写盘

最大允许 500 MB，但实现将完整文件读入一个 bytes 对象，再同步写盘。风险包括：

- 单请求约 500 MB 以上的内存峰值；
- 多并发上传造成 OOM；
- 同步磁盘写操作阻塞事件循环；
- 摘要只能在完整读取后得到；
- 声明大小不可靠时，超限文件要到完整读取后才拒绝。

### 5.5 P1：文件先落盘，任务拒绝后产生不可见孤儿文件

当前先持久化制品，再检查同包锁和全局并发限制。若准入失败：

- `<digest>.artifact` 已存在；
- metadata 已存在；
- 没有对应 BuildTask；
- 未注册包接口无法列出；
- 现有 API 无法清理。

同 digest 重复上传时还可能覆盖原始文件名 metadata，导致审计展示不稳定。

### 5.6 P1：并发准入不是数据库级原子操作

`try_insert()` 使用普通的“查询同路径活动任务 → 统计全局活动任务 → 插入”流程。多个请求或多个 worker 可同时通过检查：

- 同一 digest 可能出现多个活动任务；
- 全局活动数可能突破限制；
- 应用层检查无法替代唯一约束、条件更新或行锁。

### 5.7 P1：按路径差集推导未注册包会隐藏失败任务

如果相同 digest 的文件过去已经成功注册，后续使用同一文件再次发布但失败，其路径仍存在于 `agent_registrations` 中，因此新失败任务会从未注册列表中被过滤。

根因是把以下概念混成一个路径：

- 内容身份；
- 当前待处理包；
- 一次发布尝试；
- 已注册卡片。

### 5.8 P1：构建成功与注册成功没有独立状态

当前只有 `pending/building/done/failed`。构建成功、注册失败最终也记为 `failed`，丢失了阶段语义。

因此：

- 无法判断是否已经存在可用镜像；
- 重试只能重新构建；
- 无法按 factory/registry/cleanup 分类统计；
- 用户看到的错误语义不精确。

### 5.9 P1：管理员卡片分页总数被实例数覆盖

卡片列表先从注册中心得到分页 `total`，之后管理员投影循环又把同名变量 `total` 用作单卡实例总数，最终响应的分页总数变为最后一张卡片的实例数。

这会导致上传后刷新页面时分页器总数错误、消失或页数变化。该问题与数据库迁移相关，但本质是独立的实现 bug。

### 5.10 P1：卡片存在双事实源

当前卡片列表和详情依赖注册中心，再用 `agent_registrations` 补齐 package path 和 agent name。

风险包括：

- 任一数据源不可用都会影响完整展示；
- 两边更新顺序产生短暂或永久不一致；
- 删除时资源定位依赖本地补充记录；
- 无法明确哪个系统负责卡片字段迁移；
- 注册中心已有记录但本地无记录时，清理不完整。

### 5.11 P1：启动命令被复用为 framework

启动命令可能包含参数或随版本变化，不适合承担稳定身份。当前卡片主键实际受管理员输入的命令影响，会导致：

- 同一 Agent 因命令差异成为不同 framework；
- framework 无法稳定用于筛选和实例关联；
- 未来 TUI/Web 多接入模式无法表达；
- 重试时必须重新输入，且可能与原值不同。

### 5.12 P2：前端把状态查询异常当成流程结束

前端轮询发生任何异常都会停止轮询、关闭窗口且不展示错误原因。短暂网络波动时，用户无法区分：

- 上架成功；
- 构建仍在继续；
- 状态暂不可用；
- 后端任务已经失败。

### 5.13 P2：删除和清理缺少阶段记录与幂等重试

卡片删除跨越：

- 实例检查；
- 本地 package 删除；
- archive 删除；
- 已加载 Docker 镜像删除；
- 注册中心记录删除；
- 本地注册补充记录删除。

这些操作不可能共享事务。当前没有 cleanup 状态和对账任务，部分失败后只能依赖日志。Docker 镜像删除失败只记录 warning，接口仍可能返回成功。

### 5.14 P2：本地清理缺少统一目录归属校验

部分 archive 路径会通过配置根目录推导和校验，但通用 `LocalFileCleaner` 接收字符串后直接按路径删除，没有统一验证目标必须位于 package/image 配置目录下。

正常业务数据来源相对可信，但从纵深防御角度，应在所有本地删除入口统一执行 resolve、根目录包含关系和文件类型校验。

### 5.15 P2：错误信息缺少稳定错误码

当前主要保存异常字符串，最多 1024 字符。前端只能展示文本，无法稳定区分：

- 上传门禁失败；
- 平台不匹配；
- 基础镜像缺失；
- Docker 构建失败；
- 注册中心不可用；
- 注册冲突；
- 清理失败。

这也不利于监控聚合、告警和自动重试策略。

## 6. 新设计的目标数据边界

最新设计将数据职责调整为：

```text
注册中心 ImageEntry
  = 已注册卡片当前状态的唯一事实源

管理面 local_agent_packages
  = 尚未注册成功的软件包及当前处理状态

管理面 agent_build_history
  = 每次构建、注册或清理尝试的历史事实
```

### 6.1 注册中心 ImageEntry

稳定主键改为 `name + version`。注册中心完整保存：

- 展示信息；
- 默认版本；
- `access_mode`；
- runtime spec；
- 环境、工作区和挂载；
- 镜像引用和模块版本；
- package path 和 image archive path；
- 上传人与创建时间。

`framework` 不再承载启动命令，只作为普通展示或分类字段。

### 6.2 local_agent_packages

只保存尚未完成注册的软件包，建议核心字段包括：

- `content_digest`：内容主键和锁键；
- `package_path`：受控目录中的服务端路径；
- `original_filename`、`size_bytes`、`uploaded_by`；
- `access_mode`：完整接入配置；
- `state`：`uploaded/building/build_failed/registering/register_failed`；
- `locked_by`、`locked_until`：带租约的互斥锁；
- `last_error_code`、`last_error_message`；
- `created_at`、`updated_at`。

注册中心回读确认成功后，删除的是这条“未注册包业务记录”。源文件是否物理删除应另设清晰的保留策略。

### 6.3 agent_build_history

每次尝试新增一条，不参与当前卡片投影。建议保存：

- request ID、digest、操作类型；
- 阶段、进度和发起人；
- access mode 快照；
- 制品 name/version；
- 镜像引用、摘要、归档路径；
- Recipe 和基础镜像；
- 注册后的 name/version 快照；
- 错误阶段、错误码和错误详情；
- 开始、完成时间。

这张表回答“过去发生了什么”，而不是“现在有哪些卡片或未注册包”。

## 7. 新设计可以解决或显著缓解的问题

| 当前问题 | 新设计机制 | 预期结果 |
|---|---|---|
| 未注册包依靠表间差集 | 显式 `local_agent_packages` | 失败包稳定可见，不受历史注册影响 |
| 活动状态永久充当锁 | `locked_by + locked_until` | 支持锁租约、超时回收和重启恢复 |
| 应用层非原子查后插 | digest 主键、条件更新或行锁 | 同摘要互斥可由数据库原子保证 |
| 构建与注册失败混为一类 | 包 state 与 history phase 分离 | 可区分 factory 和 registry 失败 |
| 注册失败必须重新构建 | 保存完整构建结果与 `register_failed` | 可只重试注册 |
| 重试丢失启动命令 | 包表保存完整 `access_mode` | 重试沿用或显式修改完整配置 |
| 卡片双事实源 | 注册中心 ImageEntry 唯一事实源 | 删除 local merge 和字段不一致窗口 |
| framework 承载启动命令 | `name + version` 主键，结构化 access mode | 身份稳定并支持 TUI/Web |
| 构建历史污染当前状态 | history 明确禁止参与投影 | 当前状态与审计历史解耦 |
| 删除默认版本多端计算 | 默认版本规则归注册中心 | 减少前后端竞态与排序差异 |
| 错误只有字符串 | error stage/code/message | 支持稳定展示、统计和重试策略 |
| 进程重启后无法判断任务 | `updated_at`、锁租约、启动修复 | 可将中断任务转失败或恢复处理 |

## 8. 新设计不能自动解决的问题

数据库改造是必要条件，但不是完整解决方案。以下问题必须在业务代码和接口契约中单独治理。

### 8.1 上传仍需改为流式处理

推荐流程：

```text
创建唯一临时文件
→ 分块读取 UploadFile
→ 每块累计实际大小并更新 SHA-256
→ 超限立即终止并清理临时文件
→ fsync/close
→ 检查磁盘与最终摘要
→ 原子移动到 digest 路径
→ 在数据库事务中创建或合并 local package 记录
```

需要考虑同 digest 并发上传、临时文件唯一命名、文件已存在时的幂等行为，以及 metadata 写库后不再依赖 sidecar JSON。

### 8.2 必须定义远端任务丢失和超时策略

建议区分：

- 网络超时：指数退避、有限次数重试；
- 5xx：可重试错误；
- 404：短宽限期内重试，超过宽限期视为远端任务丢失；
- 4xx：不可重试协议/请求错误；
- 构建总时长超过上限：标记本次 history 失败并释放租约。

### 8.3 跨组件一致性依靠业务键、状态机和补偿

注册中心、管理面数据库、文件系统和 Docker 无法组成同一个 ACID 事务，但当前业务已经具备足够的权威身份字段，不需要额外设计一套端到端 `operation_id` 才能实现注册幂等：

| 范围 | 权威字段 | 职责 |
|---|---|---|
| 本地上传包 | `content_digest` | 标识相同源制品、去重并作为本地锁键 |
| 注册中心卡片 | `name + version` | 联合主键，标识唯一卡片版本并保证注册幂等 |
| 本地资源 | `package_path` | 定位和清理文件，不作为注册中心主键 |
| 一次构建尝试 | `request_id` | 构建历史主键、日志关联和锁 owner，不作为卡片身份 |

注册中心需要明确同一 `name + version` 重复提交的行为：

1. 主键不存在：创建卡片；
2. 主键已存在且关键内容等价：返回已有卡片，视为幂等成功；
3. 主键已存在但关键内容不同：返回 409，禁止静默覆盖。

关键内容至少应包含最终镜像身份和运行行为，例如 `image_digest`、`runtime_spec.rootfs.imageurl`、`access_mode` 和 `image_module_version`。如果新版注册中心保存 `content_digest`，也应参与比较。`package_path` 只表示部署环境中的本地位置，路径变化不应把同一卡片变成新的业务对象。

在注册请求超时或响应丢失时，管理面不能直接认定失败，也不需要生成新的幂等键。正确处理是按已经由工厂解析出的 `name + version` 回读注册中心：

```text
注册请求超时或本地完成状态写入失败
→ GET 注册中心 name + version
→ 卡片存在且关键内容一致：本地补记 completed
→ 卡片不存在：安全重试同一注册请求
→ 卡片存在但关键内容不同：记录 conflict，不自动覆盖
→ 注册中心仍不可用：保持 registering/unknown，稍后对账
```

因此，这部分的主要缺口不是“缺少权威幂等字段”，而是缺少围绕现有业务键的行为契约和恢复逻辑。实现应具备：

- 注册中心同主键重复请求的等价比较和 409 冲突语义；
- 注册超时或本地落库失败后的 `name + version` 回读确认；
- 构建结果和每个阶段的本地持久化；
- 清理操作幂等；
- 定时扫描 `registering`、过期锁和 cleanup 失败；
- 可人工触发的对账和重试接口。

在当前“同版本禁止覆盖”的业务约束下，这部分风险为中低风险。额外 `operation_id` 只有在未来允许同一卡片多次更新、创建时尚不知道 `name/version`，或一个操作同时创建多个注册对象时才有明显增益，不是本轮迁移的前置条件。

### 8.4 image-process 内存任务需要明确契约

允许 image-process 只保留实时内存任务，但必须明确：

- 管理面数据库是持久历史；
- 管理面何时同步最终结果；
- image-process 重启后同 request ID 是否允许重新提交；
- 重复提交是返回已有结果、重新执行还是拒绝；
- 已产生镜像但响应丢失时如何识别和恢复。

### 8.5 分页变量、前端轮询和路径校验仍是独立修复项

这些属于实现缺陷，不应等待数据库迁移自然消失：

- 修复卡片分页总数变量覆盖；
- 前端轮询错误不应直接等价于完成；
- 本地删除统一做根目录归属校验；
- 错误响应输出结构化 code/stage；
- 前端保存并展示 request ID。

## 9. 推荐目标状态机

### 9.1 包状态

```text
uploaded
  ├─ 获取锁成功 → building
  └─ 删除 → 本地记录与文件清理

building
  ├─ 构建成功 → registering
  ├─ 构建失败 → build_failed
  └─ 任务超时/进程中断 → build_failed

build_failed
  ├─ 重试构建 → building
  └─ 删除 → 清理

registering
  ├─ 注册并回读成功 → 删除 local package 记录
  ├─ 注册失败 → register_failed
  └─ 状态不确定 → 保持并交给对账任务

register_failed
  ├─ 只重试注册 → registering
  ├─ 显式重新构建 → building
  └─ 删除 → 清理已有镜像和包
```

### 9.2 构建历史阶段

推荐至少包含：

- `queued`
- `building`
- `registering`
- `completed`
- `failed`
- `cancelled`

每次重试生成新的 history，不覆盖过去记录。`operation` 区分：

- `publish`
- `retry`
- `register_retry`
- 后续可选 `cleanup_retry`

### 9.3 锁规则

1. 锁键固定为 `content_digest`；
2. 通过数据库条件更新或行锁获取，禁止查后插；
3. 锁必须有 `locked_until`；
4. 长构建需要续租；
5. request ID 必须与 `locked_by` 一致才能更新状态或释放锁；
6. 释放锁使用带 owner 条件的更新，防止旧任务释放新任务的锁；
7. 服务启动和定时任务都扫描过期锁；
8. 过期锁对应 history 应落为明确的 interrupted/timeout 失败。

## 10. 推荐的新上架流程

```text
管理员上传文件 + 完整 access_mode
→ 流式门禁、摘要和原子落盘
→ upsert local_agent_packages(state=uploaded)
→ 创建 agent_build_history(phase=queued)
→ 数据库按 digest 原子抢占租约
→ history=building，package=building
→ image-process buildFromPath(path, options, request_id)
→ 持续同步进度并续租
→ 工厂返回 name/version 和构建结果
→ 持久化所有结果快照
→ package=registering，history=registering
→ 注册中心按 name+version 幂等登记
→ 按 name+version 回读并确认关键内容一致
→ history=completed
→ 删除 local_agent_packages 业务记录
→ 按策略保留或清理源文件
→ 释放锁
```

失败处理：

- 门禁失败：不创建正式 package；清理临时文件；返回同步 4xx/507；
- 构建失败：保留 package，置 `build_failed`，history 记录 `error_stage=factory`；
- 注册失败：保留 package 和构建结果，置 `register_failed`；
- 状态不确定：先按 `name + version` 回读，不要立即重复构建；
- 数据库更新失败：利用已持久化的构建结果和注册中心回读恢复；
- 清理失败：业务成功与资源清理状态分别记录，后台幂等重试。

## 11. 推荐错误模型

API 和数据库应至少统一以下字段：

```json
{
  "stage": "factory",
  "code": "BUILD_PLATFORM_MISMATCH",
  "message": "artifact platform linux-x64-gnu does not match linux-arm64-gnu",
  "retryable": false,
  "request_id": "build-..."
}
```

建议错误阶段：

- `gate`
- `storage`
- `lock`
- `factory_submit`
- `factory_poll`
- `factory`
- `registry`
- `database`
- `cleanup`

建议保留内部详细日志，但避免将 Docker 命令输出、内部路径或注册中心响应不加裁剪地返回普通用户。

## 12. 数据库实现注意事项

### 12.1 local_agent_packages

建议约束和索引：

- `content_digest` 主键，并校验为 64 位小写十六进制；
- `package_path` 唯一；
- `size_bytes >= 0`；
- `state` 使用数据库约束或受控枚举；
- 索引 `state, updated_at`，便于恢复扫描；
- 索引 `locked_until`；
- JSON access mode 在应用层做 schema 校验；
- 原始文件名仅用于展示，不参与路径生成。

### 12.2 agent_build_history

建议约束和索引：

- request ID 主键；
- digest 普通索引，不强依赖 package 外键，因为成功后 package 记录会删除；
- 索引 `phase, created_at`；
- 索引 `content_digest, created_at`；
- `progress` 限制为 0～100；
- path 使用 snapshot 语义，不能假设文件永久存在；
- 所有重要输入和输出使用快照，避免后续 package/card 更新改变历史解释。

### 12.3 已注册卡片

管理面不再复制卡片。注册中心应明确：

- `name + version` 唯一；
- POST 对同主键同内容返回已有结果、不同内容返回 409；
- 同版本不同 image/content digest 禁止静默覆盖；
- 默认版本切换是否在注册中心事务内完成；
- 删除卡片时实例检查和记录删除是否原子；
- package/image archive 路径是否仅管理员可见；
- access mode 的唯一性和端口格式；
- 注册回读应返回足够的关键字段供管理面判断内容等价。

## 13. 迁移方案

最新设计采用停机升级，不做运行期双写双读，这是合理选择。

### 13.1 升级顺序

```text
备份管理面数据库和本地制品目录
→ 停止管理面写入与后台任务
→ 升级注册中心及其数据
→ 验证新版注册中心 OpenAPI 和卡片完整性
→ 创建管理面新表
→ 执行管理面离线迁移脚本
→ 部署新版管理面和前端
→ 执行数据一致性检查与端到端验收
→ 恢复业务
```

### 13.2 旧表迁移

`build_tasks` → `agent_build_history`：

- 一条旧任务映射为一条历史；
- 保留 request ID、路径、状态、进度、镜像信息、错误和时间；
- 旧 `done` 只有在注册中心能确认卡片时才能解释为完整发布成功；
- 无法判断错误阶段时使用迁移专用 code，不能猜测；
- 对活动任务统一按停机时间和远端状态判断，无法恢复的标记 interrupted/failed。

生成 `local_agent_packages`：

- 对每个仍存在的本地文件计算或验证 digest；
- 查询新版注册中心判断是否已经注册；
- 仅未注册成功的文件生成 package 记录；
- 从 sidecar metadata 恢复原始文件名；
- access mode 无法从旧数据可靠恢复时，标记需管理员确认；
- 不伪造 digest、name、version 或启动命令。

`agent_registrations`：

- 只作为迁移输入；
- 卡片字段以升级后的注册中心为准；
- 新版管理面不再读取；
- 首次升级不物理删除，保留为只读备份；
- 稳定运行一个版本周期后再清理。

### 13.3 迁移脚本要求

- 可重复执行或带明确 migration version；
- 输出迁移、跳过、冲突、缺文件和需人工处理数量；
- 支持 dry-run；
- 不直接修改注册中心数据库；
- 所有卡片确认通过注册中心 API；
- 失败时能够回滚新表或从备份恢复；
- 不在新旧程序同时运行时执行。

## 14. 分阶段开发计划

### 阶段 0：冻结契约

1. 固定卡片主键为 `name + version`；
2. 固定 `access_mode` schema，TUI 默认端口为字符串 `2222`；
3. 明确注册 POST 基于 `name + version` 的幂等与 409 冲突语义；
4. 明确回读、默认版本、删除和实例查询接口；
5. 明确 package/image archive 路径的安全责任；
6. 明确同 name/version 不同 digest 的处理规则；
7. 明确 image-process 对重复 request ID 的行为。

完成标准：接口契约版本化，管理面、注册中心和镜像工厂共同确认。

### 阶段 1：先修复与迁移无关的高风险问题

1. 修复管理员卡片分页 total 覆盖；
2. 为当前远端轮询增加总超时和错误分类；
3. 启动时将旧的 pending/building 僵尸任务转为失败；
4. 前端轮询异常改为可恢复状态；
5. 本地文件删除增加目录归属校验。

这一阶段用于降低迁移开发期间的现网风险，不继续扩展旧表能力。

### 阶段 2：实现新本地数据库

1. 创建 `local_agent_packages`；
2. 创建 `agent_build_history`；
3. 增加约束和索引；
4. 实现原子摘要锁；
5. 实现租约续期和 owner 校验释放；
6. 实现启动恢复与定时过期锁扫描；
7. 实现状态转换守卫，禁止非法跃迁。

### 阶段 3：重构上传与构建编排

1. 上传改为分块流式摘要和写盘；
2. 原始文件名、大小和 access mode 写入数据库；
3. 每次操作创建独立 history；
4. 提交 image-process 时使用稳定 request ID；
5. 轮询过程中同步进度、结果并续租；
6. 构建结果先持久化，再进入 registering；
7. 工厂错误映射为结构化 code/stage。

### 阶段 4：切换新版注册中心

1. 使用 `name + version`；
2. framework 不再保存启动命令；
3. 上传提交完整 access mode；
4. 注册请求一次写全卡片数据；
5. 注册后回读确认；
6. 注册失败进入 `register_failed`；
7. 实现只重试注册；
8. 卡片查询完全回源注册中心；
9. 删除 `_merge_local_card()` 和新代码对 `AgentRegistration` 的依赖。

### 阶段 5：前端改造

1. 路由、key 和操作参数切到 `name + version`；
2. 上传表单支持完整 TUI/Web access mode；
3. 未注册列表直接展示 package state；
4. 显示构建与注册的不同阶段；
5. 保留 request ID 并允许恢复轮询；
6. 网络错误展示“状态暂不可用”，不冒充成功；
7. register_failed 提供“重试注册”，build_failed 提供“重新构建”；
8. 可按需增加历史详情与错误复制功能。

### 阶段 6：删除和对账

1. 将删除改为显式受控清理状态机；
2. 所有文件路径执行根目录归属验证；
3. Docker remove、文件删除和注册中心删除均保持幂等；
4. 记录 cleanup 错误；
5. 增加定期对账任务；
6. 提供管理员人工重试入口；
7. 默认版本规则完全交给注册中心。

### 阶段 7：离线迁移与上线

1. 完成 dry-run 和迁移报告；
2. 在生产数据副本演练；
3. 测试回滚；
4. 按停机升级顺序执行；
5. 上线后校验注册中心卡片、本地包、历史和实际文件数量；
6. 观察一个稳定周期后再删除旧表。

### 阶段 8：后续扩展

1. 对 OCI 等无法稳定识别名称的制品增加 preflight/inspectFromPath；
2. 支持更多 Recipe；
3. 增加构建取消；
4. 增加清理保留策略和容量治理；
5. 组件间通信按设计升级为 TLS/mTLS；
6. 增加构建队列、配额和可观测性。

## 15. 测试计划

### 15.1 上传门禁

- 空文件；
- 超限文件，包含 Content-Length 缺失或伪造；
- 恰好达到限制；
- 危险文件名和 Unicode 文件名；
- 流式上传内存峰值；
- 上传中断后的临时文件清理；
- 磁盘空间不足；
- 同 digest 同时上传；
- 同 digest 不同原始文件名；
- 数据库写失败后的文件补偿。

### 15.2 锁与并发

- 同 digest 只能有一个 holder；
- 不同 digest 可并行；
- 并发请求不能突破限制；
- 旧 owner 不能释放新 owner 的锁；
- 锁续期；
- 锁超时回收；
- 管理面重启恢复；
- 多 worker 和实际目标数据库并发测试。

### 15.3 构建

- 成功；
- Recipe 无匹配；
- 多 Recipe 冲突；
- 平台、架构和 libc 不匹配；
- package.json 非法；
- base image 缺失；
- docker build/save/inspect 失败；
- image-process 网络超时、5xx、404 和重启；
- 总超时；
- 同 request ID 重复提交。

### 15.4 注册

- 正常注册并回读；
- 注册中心超时但实际已写入；
- 注册 4xx/409/5xx；
- 注册成功后本地提交失败；
- 只重试注册；
- 同 name/version 同 digest 幂等；
- 同 name/version 不同 digest 冲突；
- TUI、Web 和组合 access mode；
- 注册中心不可用期间状态保持。

### 15.5 查询与前端

- 卡片只读取注册中心；
- 普通用户字段裁剪；
- 管理员实例计数；
- 实例接口失败显示未知而不是零；
- 分页总数不被实例数覆盖；
- 未注册包按显式表查询；
- history 不参与当前投影；
- 页面刷新、退出再进入后恢复进度；
- 轮询网络抖动后继续；
- 失败阶段对应正确按钮。

### 15.6 删除与清理

- 有实例时完全不改资源；
- 无实例时完整删除；
- 每个清理步骤单独失败；
- 重复删除幂等；
- 非配置目录路径被拒绝；
- symlink 和路径穿越；
- 默认版本提升由注册中心保证；
- 对账任务修复部分清理状态。

### 15.7 迁移

- 空库；
- 全部成功的历史数据；
- 构建失败和注册失败；
- pending/building 僵尸任务；
- 文件缺失；
- metadata 损坏；
- 同路径多历史；
- 同 digest 多路径；
- 注册中心已有/缺失/冲突；
- dry-run；
- 重复执行；
- 中途失败与回滚。

## 16. 可观测性与运维建议

### 16.1 指标

建议至少提供：

- 上传数量、字节数、拒绝数；
- gate/factory/registry/cleanup 分阶段失败数；
- 当前各 package state 数量；
- 当前有效锁和过期锁数量；
- 构建队列长度和运行数；
- 构建、注册耗时分布；
- image-process 轮询错误数；
- 对账发现和修复数量；
- 孤儿文件、孤儿镜像、孤儿注册记录数量。

### 16.2 日志

所有组件统一携带：

- request ID；
- content digest；
- artifact name/version（识别后）；
- registry name/version（注册阶段）；
- stage、error code；
- lock owner。

日志中不直接打印上传内容、凭据和未经裁剪的内部服务响应。

### 16.3 管理能力

建议提供受权限保护的：

- 任务详情和历史；
- 过期锁查看/回收；
- 重试构建；
- 只重试注册；
- 重试清理；
- 对账 dry-run 和执行；
- 孤儿资源报告。

## 17. 验收标准

数据库和业务迁移完成至少应满足：

1. 删除 `agent_registrations` 后，卡片列表、详情、默认版本、启动和删除不受影响；
2. 新代码不通过 `build_tasks - agent_registrations` 差集计算未注册包；
3. 同 digest 在多请求、多 worker 下只能有一个有效锁；
4. 管理面或 image-process 重启后，不存在永久 `pending/building`；
5. 所有构建都有总超时，远端 404/5xx/网络错误有确定结果；
6. 构建成功、注册失败时，可不重新构建而只重试注册；
7. 注册成功必须经过回读确认，状态不确定时进入对账而非盲目重复；
8. 500 MB 文件上传不会将完整内容读入 Python 内存；
9. 任务准入失败不会产生不可见孤儿文件；
10. 卡片主键和所有操作使用 `name + version`；
11. `framework` 不再保存启动命令；
12. 完整 `access_mode` 能在上传、失败保留、重试和注册间无损传递；
13. 管理员卡片分页总数正确；
14. 实例服务异常不会被投影为实例数零；
15. 所有本地删除目标都通过配置根目录归属校验；
16. 迁移脚本支持 dry-run、重复执行、冲突报告和回滚；
17. 已注册卡片、本地未注册包、构建历史和实际文件能通过对账验证。

## 18. 风险与待确认事项

开发前需要产品、管理面、注册中心和镜像工厂共同确认：

1. 同一 digest 是否允许发布为多个 name/version；
2. 同一 name/version 上传不同 digest 是拒绝、覆盖还是显式升级；
3. 注册成功后源制品文件是否保留，保留多久；
4. `package_path` 是否应该存入注册中心，或仅保存受控资源 ID；
5. image archive 的生命周期与卡片生命周期是否完全一致；
6. 构建并发限制是全局、每管理员、每节点还是每 Recipe；
7. 最大构建时长和锁租约时长；
8. image-process request ID 的幂等保证；
9. 注册中心同主键内容等价规则和回读一致性保证；
10. access mode 是否允许管理员在 retry 时修改；
11. 历史保留期限和错误日志脱敏规则；
12. 清理失败是否影响卡片删除 API 的业务成功语义。

## 19. 总结

当前实现已完成从文件上传、通用门禁、Recipe 构建到注册中心登记的基本闭环，Recipe 将制品类型校验从管理面移入镜像工厂的方向是正确的。但当前数据库模型仍是过渡模型：`build_tasks` 承担过多职责，未注册包依靠差集推导，卡片存在本地与注册中心双事实源，且缺少锁租约、阶段状态、恢复和补偿机制。

最新数据库设计能够从根本上解决未注册包身份、摘要互斥、构建/注册阶段区分、只重试注册、完整 access mode 保留以及卡片单一事实源等问题。现有 `content_digest` 与注册中心 `name + version` 已足以分别承担本地制品和卡片的业务幂等身份，不需要额外 `operation_id` 作为本轮前置条件。与此同时，流式上传、远端轮询超时、同主键内容等价与冲突语义、注册后回读、image-process 重启语义、路径安全、前端错误恢复和分页 bug 仍需要单独实现。

因此推荐策略不是继续扩展旧 `build_tasks + agent_registrations`，而是：先修复少量与迁移无关的高风险缺陷，随后按 `ImageEntry + local_agent_packages + agent_build_history` 的目标边界实施停机迁移，并用显式状态机、租约锁、结构化错误和对账任务完成端到端可靠性闭环。

## Knowledge Extraction

- [ ] 实施验证后提炼文件上传状态机、租约锁与跨组件补偿的通用机制。
