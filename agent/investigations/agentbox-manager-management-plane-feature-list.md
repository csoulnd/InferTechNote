---
title: "AgentBox-Manager 管理面功能清单"
type: reference
domain: agent
status: active
---

# AgentBox-Manager 管理面功能清单

> **基线**：`master` 分支最新提交 `5a9535e`（`fix(node-service): 推理容器管理优化`）。
> 本次盘点已确认：工作区代码（`backend/`、`frontend/`、`node_service/`、`sandbox-manager/`、`deploy/`）与 `master` 一致，差异仅在 `docs/` 与 `deploy/README.md` 的文档组织。
> **方法**：按前端页面 / 后端接口与服务 / 节点侧服务与部署栈三条线逐文件核对，只记录代码与配置中确认存在的能力；未实现、未接线、口径不一致项单列于第七章。
> **范围**：AgentBox-Manager 管理面（含管理面自带的节点侧代理与三方 Agent 镜像构建服务），以及部署栈提供、界面仅跳转或展示的配套能力。

---

## 一、总览

### 1.1 产品定位与角色模型

AgentBox（灵昭）AI 一体机随箱交付的管理控制台，面向两类使用者：

| 角色 | 工作区 | 可用范围 |
|------|--------|----------|
| 管理员 `admin` | 管理视图 / 个人视图（可切换） | 全部资源：整机硬件、模型与推理服务、用户与权限、日志与审计、智能体与技能发布 |
| 普通用户 `user` | 仅个人视图 | 自助能力：API Key 申请与撤销、个人用量、技能安装、个人信息与云账户 |

- 顶栏「管理视图 / 个人视图」切换（仅 admin 可见），前端以 `effectiveIsAdmin = isAdmin && workspace !== 'user'` 作为管理面判据（`composables/useAuth.ts`、`layout/TopNav.vue`）。
- 路由级 `adminOnly` / `userOnly` 名单双重约束，越权跳 `/403`（`router/index.ts`、`router/menu.ts`）。

### 1.2 能力域分布

| 能力域 | 功能点数 | 面向角色 | 关键外部依赖 |
|--------|---------|----------|--------------|
| A 身份认证与账号安全 | 6 | 全部 | PostgreSQL、CCCollectSDK（UDS） |
| B 用户管理（管理面） | 7 | admin | PostgreSQL、LiteLLM、用户家目录 FS |
| C 个人中心 | 5 | 全部 | PostgreSQL、CCCollectSDK |
| D 一体机硬件可观测 | 5 | admin | VictoriaMetrics、node/npu-exporter |
| E 推理模型与 MaaS（模型监控） | 12 | admin + user（只读） | LiteLLM、PostgreSQL、VictoriaMetrics、Grafana |
| F API Key 与调用接入 | 5 | user（admin 可查） | LiteLLM、PostgreSQL |
| G 推理服务节点部署与管理 | 11 | admin | 各节点 agentos-node-service、Docker |
| H 智能体实例监控 | 4 | admin | Agent 注册中心 |
| I 智能体配置与配置下发 | 5 | admin 可写 / 全员可读 | etcd、本地权威 config |
| J 三方智能体管理与镜像构建 | 8 | admin 可写 / 全员可读 | image-process、Docker、Agent 注册中心 |
| K 技能库 | 4 | admin + user | SkillHub |
| L 日志中心与任务中心 | 9 | admin | Loki、Alloy、Grafana、CCAgent SDK |
| M 平台横切能力 | 8 | 全部 | 事件总线、审计日志、nginx |
| N 部署与安装配套 | 8 | 运维 | Docker Compose、宿主机 systemd、ascend-deployer |
| **合计** | **97** | — | — |

### 1.3 菜单 → 页面 → 能力域映射

| 一级菜单 | 二级菜单（路由名） | 可见性 | 对应章节 |
|----------|-------------------|--------|----------|
| 资源管理 | 一体机 `appliance` | adminOnly | D |
| 资源管理 | 推理模型 › 模型监控 `inference-model-dashboard` | 全员 | E |
| 资源管理 | 推理模型 › API 接入 `inference-model-api-key` | userOnly | F |
| 资源管理 | 推理模型 › 推理模型调用分析 `inference-model-call-analysis` | adminOnly（隐藏菜单） | E |
| 资源管理 | 推理模型 › 推理服务节点 `node-service` | adminOnly | G |
| 资源管理 | 智能体 › 智能体监控 `agent-monitor` | adminOnly | H |
| 资源管理 | 智能体 › 智能体配置 `agent-config` | 全员（写权限限 admin） | I |
| 资源管理 | 智能体 › 三方智能体管理 `agent-framework` | 全员（写权限限 admin） | J |
| 资源管理 | 技能库 | 全员（外链 `:8098`） | K |
| 系统设置 | 用户管理 `user-management` | adminOnly | B |
| 系统设置 | 日志中心 `log-center` | adminOnly | L |
| 系统设置 | 日志预览 `log-explore` | adminOnly（隐藏菜单） | L |
| 顶栏入口 | 任务中心 `task-center` | 全员（隐藏菜单） | L |
| 顶栏入口 | 个人中心 `profile` | 全员（隐藏菜单） | C |

---

## 二、功能清单

### A. 身份认证与账号安全

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| A1 | 本地账号密码登录 | 校验本地 `user` 表；签发 JWT 对（access 默认 15 分钟 / refresh 默认 2 天，HS256），写入 HttpOnly + SameSite=lax Cookie，前端不接触 token | `POST /api/v1/auth/login` | 免鉴权 |
| A2 | Token 刷新与保活 | 刷新时校验 `token_version` 与 `revoked_after`；前端 401 单飞刷新后重放原请求，`App.vue` 每 5 分钟主动保活 | `POST /api/v1/auth/refresh` | 免鉴权 |
| A3 | 登出与令牌撤销 | 登出、改密、管理员重置密码均使既有令牌失效（`user_revocation.revoked_after` / `user.token_version`） | `POST /api/v1/auth/logout` | 登录即可 |
| A4 | 资源级权限矩阵 | 硬编码零 IO 的角色-资源-动作矩阵（read/write/delete/manage），`admin = *`，`user` 仅授予 dashboard/api_keys/usage/apps/alerts/skills/agentos_config 子集 | `GET /api/v1/auth/permissions`、`POST /api/v1/auth/verify` | 登录即可 |
| A5 | 第三方应用单点登录（OAuth2 Provider） | 授权码模式：consent 页允许/拒绝 + 「记住授权」（localStorage），授权码一次性换取独立类型的 `oauth2_access` JWT（默认 1440 分钟）；仅当 `OAUTH2_CLIENT_ID/SECRET/NAME` 齐备时才注册路由 | `GET|POST /api/v1/oauth2/authorize`、`POST /api/v1/oauth2/token`、`GET /api/v1/oauth2/userinfo` | 见各接口 |
| A6 | 华为 ID / UniSSO 登录与云管设备绑定 | 经 CCCollectSDK UDS 通道调用云管完成 OAuth 登录、设备授权绑定/解绑；个人中心内以退避轮询（3s 起至 24s，总超时 10 分钟）跟踪绑定状态 | `GET /api/v1/unisso/auth/huawei-id/login`、`/auth/devices`、`/user/detail`、`POST /api/v1/unisso/edge/nedevices/action/bind`、`.../unbind`、`GET /bind-status` | 设备接口凭 `X-Access-Token` |

### B. 用户管理（管理面）

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| B1 | 用户列表与检索 | 分页、关键字搜索、角色筛选、排序；管理员自身行禁用重置密码与删除 | `GET /api/v1/users` | admin |
| B2 | 单条创建用户 | 建本地账号 → 同步 LiteLLM 用户 → 触发用户供给事件（家目录、配置、预装 skill）；任一环节失败按 SAGA 逆序回滚 | `POST /api/v1/users/batch` | admin |
| B3 | 批量创建（CSV） | 下载导入模板、上传 CSV、分批（每批 5，上限 100）执行并展示逐条进度与结果；下载用户凭证 | `POST /api/v1/users/batch` | admin |
| B4 | 角色与状态变更 | 修改 role / is_active | `PATCH /api/v1/users/{user_id}` | admin |
| B5 | 删除用户 | 删除本地账号并同步清理 LiteLLM 用户侧资源，强依赖 LiteLLM 可达 | `DELETE /api/v1/users/{user_id}` | admin |
| B6 | 重置密码 | 管理员重置指定用户密码，并使该用户既有令牌失效 | `POST /api/v1/users/{user_id}/reset-password` | admin |
| B7 | 预装技能异步化 | 建号后的预装 skill 改为后台异步执行（`AGENTOS_PRESET_SKILL_NAMES` 名单，源缺失静默跳过），创建接口先返回，关闭时排空任务 | 由用户创建事件驱动 | admin |

### C. 个人中心

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| C1 | 个人信息 | 展示用户名、用户 ID、角色标签 | `GET /api/v1/users/me` | 登录即可 |
| C2 | 修改密码 | 前端强校验：8–64 位、至少 2 类字符、不得包含用户名 | `PUT /api/v1/users/me/password` | 登录即可 |
| C3 | 云账户管理 | 华为云账号绑定/解绑状态、绑定引导 | UniSSO 接口组（见 A6） | admin 显示该 Tab |
| C4 | 偏好设置 | 页面占位，暂无实际配置项 | — | — |
| C5 | 退出登录 | 调后端登出并清理本地状态，跳登录页 | `POST /api/v1/auth/logout` | 登录即可 |

### D. 一体机硬件可观测

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| D1 | 节点清单 | 由 `NODE_EXPORTER_HOST` + `WORKER_NODES` 推导 `master` / `worker-N`，多机集群同一视图内切换（切节点写入 URL） | `GET /api/v1/hardware/nodes` | admin |
| D2 | 单节点实时快照 | CPU / 内存 / 磁盘 / 网络 / 昇腾 NPU 设备级指标卡，30s 轮询 | `GET /api/v1/hardware/snapshot?node=` | admin |
| D3 | 设备标识与运行信息 | 设备型号、运行时长、IP 标注卡 | 同上 | admin |
| D4 | 视图与详情 | 3 种设备视图切换、NPU 详情弹窗、磁盘详情弹窗、指标卡横向滚动 | 前端交互 | admin |
| D5 | 指标来源治理 | 指标由 node_exporter / npu-exporter 暴露、VictoriaMetrics 抓取与保存（默认 30 天），管理面仅做 PromQL 查询 | 部署侧 `victoriametrics/*` | — |

### E. 推理模型与 MaaS（模型监控）

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| E1 | 模型列表与详情 | 合并 LiteLLM 模型数据与面板本地扩展字段（`metrics_endpoints`、`max_concurrent` 等） | `GET /api/v1/litellm/model`、`GET /api/v1/litellm/model/{model_id}` | 登录即可（非 admin 脱敏 `api_base`、隐藏 `api_key`） |
| E2 | 添加模型 | 注册到 LiteLLM 并落库本地扩展字段；提交前做必填项、KEY 格式、`api_base` URL 校验与 `/models` 探活（`MODEL_CREATE_PROBE_ENABLED`，默认 5s 超时） | `POST /api/v1/litellm/model` | admin |
| E3 | 编辑模型 | 更新 LiteLLM 配置与本地扩展字段，补齐 `(model, api_base)` 唯一性校验并规范化比较 | `PUT /api/v1/litellm/model/{model_id}` | admin |
| E4 | 删除模型 | LiteLLM + 本地同时删除，404/400 幂等 | `DELETE /api/v1/litellm/model/{model_id}` | admin |
| E5 | 模型健康探测 | 独立探测 LiteLLM `/health`，返回 `{model_id: status}` 映射，驱动模型卡片状态 Tab | `GET /api/v1/litellm/model/health` | admin |
| E6 | 监控采集点管理 | 为模型维护推理引擎与指标地址列表（前端编辑器 + 同 URL 冲突校验）；内部 http_sd 接口聚合并去重后供 VictoriaMetrics 抓取 | `GET /internal/vm/inference-metrics`（内网、无鉴权） | 部署侧调用 |
| E7 | 推理性能可视化 | 模型详情页按监控点选择 Grafana 性能面板并 iframe 内嵌（vLLM / SGLang 两套面板），嵌入前先刷新 Cookie 会话 | 前端 `PerformanceMonitor` + `/api/v1/auth/refresh` | admin |
| E8 | 今日调用概览 | 管理面 4 张 hero 卡（token / 请求数 / 成本 / 成功率），个人视图 2 张 trend 卡 + ECharts 趋势 | `GET /api/v1/litellm/usage/overview`（×3）、`GET /api/v1/litellm/usage/user`（×3） | 按角色分支 |
| E9 | 调用分析（管理面） | 概览卡、调用趋势（预设 + 自定义日期）、按模型调用量堆叠柱、活跃用户折线、用户用量排行（TOP10 / ALL / 自定义） | `GET /api/v1/litellm/usage/overview|by-user|model-trend` | admin |
| E10 | 用量统计底座 | 直连 LiteLLM 自有库按日/模型/用户聚合（`LiteLLM_SpendLogs` 等），不走 LiteLLM HTTP API；普通用户仅可见本人数据 | `GET /api/v1/litellm/usage/` 下 `trend`、`by-model`、`model-trend`、`by-user`、`user`、`overview`、`user-model-trend` | 按角色/权限矩阵 |
| E11 | MaaS Gateway 接入信息 | 返回 MASS Gateway 地址，任意登录用户申请 API Key 后即可调用推理接口 | `GET /api/v1/maas/config` | 登录即可 |
| E12 | 使用指南 | 页面内嵌接入示例（含代码块与复制） | 前端 | 全员 |

### F. API Key 与调用接入

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| F1 | Key 列表 | 直查本地映射表（不调 LiteLLM API），含分页 | `GET /api/v1/litellm/key` | 登录即可（按 `user_id` 隔离） |
| F2 | 申请 Key | 调 LiteLLM 生成，Key 以 AES-GCM 加密落库；每用户上限 `MAX_KEYS_PER_USER=10`；名称限 `[a-zA-Z0-9_-]`、≤256 | `POST /api/v1/litellm/key/generate` | 登录即可 |
| F3 | 删除 Key | 先校验归属 → 调 LiteLLM 删除 → 清本地映射；默认 Key 前端不提供删除入口 | `DELETE /api/v1/litellm/key/{key_alias}` | 登录即可 |
| F4 | 一次性展示 | 创建成功弹窗仅展示一次完整 Key，支持复制 | 前端交互 | — |
| F5 | 接入隔离 | 「API 接入」页 `userOnly`，管理视图下不可访问（跳 403） | 路由守卫 | — |

### G. 推理服务节点部署与管理

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| G1 | 多节点代理层 | 前端不直连节点：管理面按 `node` 参数解析 `host:port`，透传调用者 JWT 转发；节点不可达映射 502、超时映射 504 | `/api/v1/node-service/*`（均带 `?node=`） | `node_service:manage`（实际仅 admin） |
| G2 | 简化配置读取/修改 | 以节点 `config/config.json` 为唯一读写源，PUT 为局部合并语义，并同步白名单字段进生效配置 | `GET|PUT /api/v1/node-service/config` | admin |
| G3 | 配置模板 | 系统模板 4 个（DeepSeek-V4-Flash A2/A3 SGLang、A3 vLLM、GLM-5.2 A3 SGLang），`user_templates/` 同名优先；模板含 `{{占位符}}` 替换（卡数、硬件类型、镜像、模型名、各端口） | `GET /api/v1/node-service/templates`、`GET /templates/{name}`、`POST /config/apply-template` | admin |
| G4 | 白名单字段约束 | 仅 8 个字段（weight_path、model_path、model_name、deploy.image、gpu_memory_utilization、max_model_len、max_num_batched_tokens、max_num_seqs）可覆盖模板，TP/DP 等引擎参数一律沿用模板 | 服务端约束 | admin |
| G5 | 服务启停 | 启动 / 停止 / 重启 / 保存整份 user_config 后重启；启动前强校验模型路径、模型名与镜像 | `POST /inference/start|stop|restart|config/save-user-config`（经管理面代理） | admin |
| G6 | 异步进度与并发保护 | 启停立即返回并后台执行；`start_progress.status ∈ queued/preparing/pulling/starting/waiting/ready/failed`，前端 10s 轮询；重入返回 409，`_start_generation` 使旧线程写入失效 | `GET /api/v1/node-service/status` | admin |
| G7 | 引擎支持与钳制 | vLLM（MindIE Motor 镜像）/ SGLang 双引擎；`allowed_engines`（默认 `["vllm"]`，可开 sglang）对启动、模板、镜像接口统一钳制 | 节点侧 `NODE_SERVICE_ALLOWED_ENGINES` | admin |
| G8 | 状态与健康 | 容器状态、实际引擎识别（mindie_motor / sglang / vllm / not_detected）、最近日志、当前配置 | `GET /status`、`GET /health` | admin |
| G9 | 容器日志查看 | 行数 1–1000，前端可开启 10s 自动刷新 | `GET /api/v1/node-service/logs` | admin |
| G10 | 镜像清单 | 按引擎过滤节点可用 Docker 镜像 | `GET /api/v1/node-service/images?engine=` | admin |
| G11 | 节点侧鉴权与操作审计 | 节点服务回查管理面 `/api/v1/auth/verify`（resource=node-service、action=manage）判定 admin；逐条 key=value 记录配置变更、启停结果、模板应用与鉴权拒绝 | 节点侧 `op_log` | — |

### H. 智能体实例监控

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| H1 | 实例概览卡 | 总数 / 运行 / 异常 / 停止 | `GET /api/v1/agent/instances` | admin |
| H2 | 实例列表 | 分页、关键字搜索、排序、框架与状态筛选、手动刷新；30s 轮询；注册中心未配置或 502/503/504 时清空并提示 | 同上 | admin |
| H3 | 注册中心对接 | 代理 Agent 注册中心 `GET /api/instances`；内存快照，仅首访或 `refresh=true` 拉取 | `AGENT_REGISTER_URL` | — |
| H4 | 可选启停 | 依赖 `AGENT_REGISTER_URL` 是否配置，未配置则功能整体不可用 | — | — |

### I. 智能体配置与配置下发

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| I1 | 沙箱资源配置 | 空闲销毁时长、CPU、内存配额 | `GET|PUT /api/v1/agentos-config` | 读：登录即可；写：admin |
| I2 | 出网（egress）策略 | 默认策略与默认域名、域名/IP 黑白名单（前端行级增删 + 域名通配与 CIDR 校验） | 同上 | 同上 |
| I3 | 乐观锁并发控制 | PUT 携带 `expected_revision`，冲突后重新拉取，避免并发覆盖 | 同上 | admin |
| I4 | 双写下发链路 | 先写 etcd 数据面 key，成功后再原子写本地权威 config；本地写盘失败则补偿回滚 etcd | 同上 | admin |
| I5 | 降级行为 | etcd 未配置时 GET 可用、PUT 返回 503；权威配置文件损坏时读写均 503 并提示 | 同上 | — |

### J. 三方智能体管理与镜像构建

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| J1 | 卡片/网格浏览 | 卡片与网格两种视图、搜索、分页、卡片详情 | `GET /api/v1/thirdparty_agent/cards`、`GET /cards/{name}/{version}` | 登录即可 |
| J2 | 接入上传（分片续传） | 创建上传会话 → 64MiB 分片上传（断点续传，进度存档于 localStorage）→ 完成；单包上限 2GiB，会话 1 小时 TTL、每 10 分钟清理 | `POST /uploads`、`PUT /uploads/{id}/parts/{n}`、`GET /uploads/{id}`、`POST /uploads/{id}/complete`、`DELETE /uploads/{id}` | admin |
| J3 | 镜像构建 | 制品入库后交由 image-process 基于宿主机 Docker 构建 OCI 镜像（`FROM agent-base:1.0`），上传与构建进度分离展示；可选 `docker save` + `gzip` 归档 | `POST /cards`、`POST /cards/from-artifact` | admin |
| J4 | 卡片发布 | 构建产物向 Agent 注册中心发布卡片（202 受理），前端 2s 轮询状态 | 同上 + `GET /unregistered/{digest}` | admin |
| J5 | 卡片治理 | 描述编辑、设为默认版本、删除（删除卡片按钮在删除过程中置灰）；删除时同步移除 Docker daemon 镜像 | `PATCH /cards/{name}/{version}`、`PUT /cards/{name}/default`、`DELETE /cards/{name}/{version}` | admin |
| J6 | 未注册制品区 | 列出未完成注册的制品、查看状态、重试注册、删除 | `GET /unregistered`、`GET /unregistered/{digest}`、`POST /unregistered/{digest}/retry`、`DELETE /unregistered/{digest}` | admin |
| J7 | 启动自愈 | 启动时恢复中断的注册流程（`recover_registering`）与中断的构建任务（`recover_interrupted_builds`，标记失败） | 启动期 | — |
| J8 | 双视图差异 | 个人视图剥离实例数、仅只读；接入/删除/设默认仅 admin 可用 | 前端 `FrameworkPage.vue` | — |

### K. 技能库

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| K1 | 技能市场入口 | 一级菜单为外链，跳转 `http(s)://<同主机>:8098`（SkillHub 前端，端口硬编码），新窗口打开 | 前端 `menu.ts` `externalUrl` | 全员 |
| K2 | 技能安装 | 从 SkillHub 下载 zip → sha256 校验 → 防 zip-slip 解压到本人技能目录 → 写 `skills_state.json`；支持指定版本与强制覆盖；SkillHub 未配置时返回 503 | `POST /api/v1/skills/install` | `skills:write`（user/admin 均有） |
| K3 | 预装技能 | 建号时按名单异步拷贝预置 skill 到用户技能目录，与 B7 同源 | 事件驱动 | — |
| K4 | 市场侧能力归属 | 技能市场容器（frontend 8098、minio 8099 对外）由部署栈独立 compose 管理，管理面不启停 | 部署侧 `skillhub/` | — |

### L. 日志中心与任务中心

| 编号 | 功能点 | 说明 | 关键接口 | 权限 |
|------|--------|------|----------|------|
| L1 | 日志分类与组件 | 分类卡片 + 组件数量；内置组件：管理面、推理服务、agent-gateway、agent-registry、agent-runtime（含 runtime-*.log 合并规则）、jiuwenbox | `GET /api/v1/logs/categories`、`GET /api/v1/logs/components` | admin |
| L2 | 文件树浏览 | 按节点切换日志文件树并逐级下钻；文件名列表来自 Loki 标签（节点 IP + 分类限定路径） | `GET /api/v1/logs/loki/filenames?ip=&category=`、`GET /api/v1/logs/components/{id}/resolve-path` | admin |
| L3 | 日志预览检索 | 关键词 + 时间范围（1h/6h/12h/24h/7d），LogQL 只读预览，结果通过 Grafana Explore iframe 呈现 | `POST /api/v1/logs/loki/query` | admin |
| L4 | 日志导出任务 | 按条件创建 Loki 导出任务，后台 worker 执行（并发 2、单文件上限 512MB、保留 7 天、每日 03:00 清理） | `POST /api/v1/logs/loki/export`、`GET /api/v1/logs/exports`、`GET /exports/{task_id}`、`GET /exports/{task_id}/download`、`DELETE /exports/{task_id}` | admin |
| L5 | 任务中心 | 顶栏入口，展示导出/打包任务（搜索、前端分页、下载 zip、删除），存在未完成任务时 2s 轮询 | 同 L4 | 登录即可 |
| L6 | 容器日志接入 | vLLM 等推理容器日志经 Alloy 从 docker daemon 采集入 Loki（无磁盘文件），与 L1 的「推理服务」组件对应 | 部署侧 `alloy/config.alloy` | — |
| L7 | 反馈问题 | 问题描述 + 图片（≤5 张、单张 ≤10MB）+ 手机号 + 故障时间（不可选未来）+ 共享日志勾选，提交后 loading/成功 3s 倒计时/失败三态 | `POST /api/v1/agent/log/submit`（multipart） | 登录即可 |
| L8 | 日志采集下发 | 经 CCAgent SDK（UDS）下发采集任务并轮询进度（间隔 3s、最长 10 分钟） | `services/cc_agent_service.py` | — |
| L9 | Grafana 直通 | 管理面为 Grafana 提供 auth_request 回查端点并注入 `X-WEBAUTH-USER/ROLE`，实现免二次登录的 iframe 内嵌 | `GET /api/v1/auth/proxy-verify` | admin |

### M. 平台横切能力

| 编号 | 功能点 | 说明 |
|------|--------|------|
| M1 | 操作审计日志 | 覆盖所有非 GET 的 `/api/*` 及任何 401/403 响应；记录语义操作码、用户、IP、方法路径、结果（success/denied/failure）、耗时、脱敏后的请求附加信息；仅落 `{LOG_DIR}/audit.log`（0600、每 30 天轮转保留 12 份），与 `app.log` 分离，异常全兜底 |
| M2 | 事件总线 | 类型订阅、顺序调用、默认异常隔离；`raise_on_error` 时抛出以驱动 SAGA 回滚。事件：`UserProvisionEvent`、`ModelChanged`（5 分钟防抖全量重建）、`StartupEvent`、`ShutdownEvent` |
| M3 | 用户供给 handler（jiuwen） | 家目录创建、config 下发、skill 拷贝、属主变更；模型变更后全量重建；启动时全量同步自愈（失败仅告警） |
| M4 | 启动期初始化与自愈 | 建表 + 轻量 ALTER 迁移、中断注册/构建恢复、初始模型注册（`INITIAL_MODELS` 或嗅探 `/models`，已存在跳过、单模型失败不阻断）、权威 config 与 etcd 一致性检查、日志调度与上传清理任务启动、管理员初始化（强依赖 LiteLLM，失败阻断启动并逆序回滚） |
| M5 | 统一接口与错误规范 | 响应统一 `ApiResponse`；前端 `ApiError` 携带 `status/code/detail`，401 单飞刷新、失败清理状态跳登录（OAuth 流程中不硬跳以保留授权参数） |
| M6 | 管理面网关行为 | nginx：SPA fallback、静态资源缓存、`/api/` 反代后端、三方 Agent 上传放宽至 2g/600s、安全响应头、`server_tokens off` |
| M7 | 前端权限与导航 | 单一数据源菜单树（`router/menu.ts`）同时生成路由与菜单；侧栏按 `adminOnly/userOnly` 过滤、折叠态持久化、外链新窗口打开 |
| M8 | 前端通用组件与组合式函数 | 轮询（跳过执行中的那一拍、卸载自停）、鉴权状态、日志、云账户状态机、反馈弹窗、代码块复制、通用下拉；模型/监控点/出网规则三类校验工具 |

### N. 部署与安装配套

| 编号 | 功能点 | 说明 |
|------|--------|------|
| N1 | 一键交付 | 一体机经 Ascend-Deployer 完成整机（含 AgentBox-Platform）安装与离线包构建；**必须先 Platform 后 Manager**，否则用户目录与预装 skill 不可用 |
| N2 | 部署脚本能力 | `deploy.sh install / up / restart / down / status / uninstall`；卸载默认保留数据，`--clean` 清理卷、`.env` 与安装目录；`up` 顺序为启动模块 → 生成硬件指标 → 嗅探模型 → compose up → skillhub → status |
| N3 | 拓扑与多机 | `--role master|worker`、`--mode single|multi`、`--workers`、`--master-ip`；worker 不运行业务 compose，指标与日志上报 master |
| N4 | 可选组件 | 模块化安装（`AGENTOS_MANAGER_MODULES`，默认 node_exporter/npu_exporter/alloy），node_service 为 opt-in（`--with-node-service`），技能市场为 opt-in（`--with-skillhub`） |
| N5 | compose 服务 | `agentos`（前端 + 后端 + nginx，8090→80）、`image-process`（内网 8091）、`postgres`（127.0.0.1:5432，agentos/litellm 双库）、`litellm`（8100→4000）、`victoriametrics`（127.0.0.1:8428）、`loki`（8096）、`grafana`（8093→3000） |
| N6 | 宿主机组件 | node_exporter（8091）、npu-exporter（8092，systemd timer，延时启动避免驱动未就绪）、alloy（12345）、agentos-node-service（8101，systemd 单元） |
| N7 | 离线部署与镜像管理 | 离线包 + `docker load`；`VERSIONS` 覆盖自建镜像 tag；镜像仅补缺失并以内置清单兜底；镜像清单与端口有独立文档 |
| N8 | 部署期钩子 | `gen_hardware_metrics.sh` 生成 VM file_sd 硬件指标目标（master + workers）；`sniff_models.sh` 嗅探推理服务并写回 `INITIAL_MODELS / API_BASE / API_KEY / INFERENCE_ENGINE / METRICS_URL`；skillhub 预置技能打包上传市场 |

---

## 三、权限与可见性矩阵

### 3.1 后端资源级权限（`backend/app/iam/permissions.py`）

| 资源 | admin | user |
|------|-------|------|
| `*`（全部） | read / write / delete / manage | — |
| `dashboard` | 同上 | read |
| `inference.api_keys` | 同上 | read / write |
| `inference.usage` | 同上 | read |
| `apps` / `alerts` | 同上 | read |
| `skills` | 同上 | read / write |
| `agentos_config` | 同上 | read |
| `inference.models` / `inference.monitor` / `agent.global` / `logs` / `hardware` / `node_service` / `settings` | 同上 | 未授予（矩阵预留） |

> 实现现状：真正走 `require_permission` 的是 `inference.usage`、`skills`、`hardware`、`node_service`、`agentos_config`；其余管理类路由使用 `require_admin`（只认 admin），普通用户类接口为「登录即可 + 按 `user_id` 隔离」。

### 3.2 页面可见性

| 页面 | 管理视图（admin） | 个人视图（admin 切换后） | 普通用户 |
|------|------------------|--------------------------|----------|
| 一体机 / 用户管理 / 日志中心 / 日志预览 / 调用分析 / 推理服务节点 / 智能体监控 | ✅ | ❌ 403 | ❌ 403 |
| 模型监控 / 智能体配置 / 三方智能体管理 | ✅（写） | ✅（读，个人视图剥离部分数据） | ✅（读） |
| API 接入（API Key） | ❌ 403（`userOnly`） | ✅ | ✅ |
| 技能库（外链）/ 任务中心 / 个人中心 | ✅ | ✅ | ✅ |

---

## 四、接口模块总览

| 模块 | 前缀 | 路由数 | 说明 |
|------|------|--------|------|
| 认证 | `/api/v1/auth` | 6 | 登录/刷新/登出/校验/权限矩阵/代理回查 |
| UniSSO | `/api/v1/unisso` | 3 | 华为 ID 登录、设备注册状态、云管用户详情 |
| 设备绑定 | `/api/v1/unisso/edge/nedevices` | 3 | 绑定/解绑/绑定状态 |
| 用户 | `/api/v1/users` | 7 | 用户 CRUD、批量创建、重置密码、本人信息与改密 |
| 推理模型 | `/api/v1/litellm/model` | 6 | 模型 CRUD、健康、详情 |
| MaaS 配置 | `/api/v1/maas` | 1 | Gateway 接入地址 |
| API Key | `/api/v1/litellm/key` | 3 | 列表/申请/删除 |
| 用量统计 | `/api/v1/litellm/usage` | 7 | 趋势、模型分布、用户维度、总览 |
| 硬件监控 | `/api/v1/hardware` | 2 | 节点列表、单节点快照 |
| 推理服务管理 | `/api/v1/node-service` | 14 | 配置、启停、状态、日志、镜像、模板、应用模板 |
| 智能体实例 | `/api/v1/agent` | 1 | 实例分页查询 |
| 智能体日志采集 | `/api/v1/agent/log` | 1 | 反馈问题日志提交 |
| 三方智能体 | `/api/v1/thirdparty_agent` | 15 | 上传分片、卡片治理、未注册制品 |
| 技能 | `/api/v1/skills` | 1 | 技能安装 |
| 日志中心 | `/api/v1/logs` | 6 | 分类、组件、路径解析、导出任务 |
| 日志中心 Loki | `/api/v1/logs/loki` | 3 | 查询、导出、文件名 |
| Agentos 配置 | `/api/v1/agentos-config` | 2 | 读取/下发配置 |
| OAuth2 Provider | `/api/v1/oauth2` | 4 | 授权、决策、令牌、用户信息（按开关注册） |
| 内部接口 | `/internal/vm` | 1 | VictoriaMetrics http_sd，仅内网 |
| 节点侧服务 | `/inference`（node_service :8101） | 13 | 节点本地配置、容器生命周期、模板 |

---

## 五、数据与外部系统依赖

| 类别 | 承载 |
|------|------|
| 管理面自有 PostgreSQL（`AGENTOS_DATABASE_URL`） | `user`、`user_revocation`、`oauth2_authorization_codes`、`litellm_model_params`、`litellm_user_key`、`user_default_key`、`log_component`、`log_export_task`、三方智能体上传会话/分片/制品/构建任务 |
| LiteLLM HTTP API | 模型注册与查询、健康探测、Key 生成与删除、用户同步 |
| LiteLLM 自有库（`LITELLM_DATABASE_URL`） | 用量统计直查（趋势、分布、排行） |
| etcd | Agentos 配置数据面下发（写入失败即报错，不静默） |
| 本地文件系统 | 权威 `agentos_config.yaml`、用户家目录与技能目录、日志与导出目录、镜像与制品目录 |
| VictoriaMetrics / Grafana | 硬件与推理性能指标存取与面板展示 |
| Loki / Alloy | 日志汇聚、检索与导出 |
| Agent 注册中心 | 智能体实例列表、三方 Agent 卡片注册 |
| CCCollectSDK（UDS）/ CCAgent SDK | 华为云账号与设备绑定、日志采集任务下发 |
| SkillHub | 技能市场前端与技能包下载 |

---

## 六、最近演进（master 近期提交对应的新增能力）

- **推理服务管理**：SGLang 引擎支持 + 配置精简 + 推理容器管理优化（含异步进度、并发保护）
- **审计与日志**：操作审计日志体系（`audit.log`）、vLLM 容器日志接入日志中心
- **登录与账号**：UniSSO / 华为 ID OAuth 登录、设备绑定与解绑、云管登录绑定
- **配置下发**：管理面下发 Agentos 配置（etcd + 权威 config 双写与补偿）、新增「智能体配置」页面
- **三方智能体**：上传与构建进度分离、分片续传、构建任务自愈、卡片删除态与默认版本治理
- **模型管理**：添加/编辑表单优化、`(model, api_base)` 唯一性校验、KEY 格式与 URL 探活
- **用户体验**：用户创建预装 skill 异步化、批量创建用户减少模型列表查询

---

## 七、边界与已知缺口（代码已确认）

1. **总览页已下线**：`overviewEnabled = false`，`OverviewPage.vue` 为占位。
2. **占位且零引用页面**：`AlarmPage.vue`、`SkillStorePage.vue` 均为占位实现，全仓无引用；技能库走外链。
3. **模板保存/删除链路不完整**：管理面代理声明 `POST /api/v1/node-service/templates` 与 `DELETE /templates/{name}`，但节点侧 `server.py` 未实现对应路由，调用将返回 404；前端页面当前仅使用模板查询接口。
4. **悬空前端 API**：`POST /api/v1/litellm/model/{id}/restart`（前端 `restartModel`）后端无对应路由；`PATCH /api/v1/users/{id}`、`GET /api/v1/logs/components`、部分用量查询方法在 api 层定义但页面无调用点。
5. **未实现指标**：调用分析页「实时并发 QPS」恒为 `--`；模型监控卡片 `todayCalls / e2eP95` 恒为 `--`、`tags` 恒为空；一体机页磁盘 IO 数据未渲染。
6. **字段可能丢失**：监控点编辑器仅提交 `{inference_engine, instance_url}`，详情页 `var-job` 依赖的 `grafana_job_name` 需后端 merge 保留。
7. **口径不一致**：个人中心「云账户管理」Tab 用 `isAdmin` 判断，路由守卫用 `effectiveIsAdmin`，admin 切到个人视图后该 Tab 仍可见。
8. **权限矩阵预留未用**：`dashboard / apps / alerts / logs / settings / inference.models / inference.monitor / agent.global` 等资源常量已定义但无路由使用；`GET /api/v1/auth/permissions` 前端无调用点。
9. **单副本约束**：image-process 构建任务表存于进程内存，三方 Agent 构建按单副本设计。
10. **无国际化**：前端未引入 i18n 框架，仅使用 Element Plus 中文语言包，界面文案为中文硬编码。
11. **无前端埋点**：无 analytics / Sentry 等上报，行为追溯依赖后端审计日志。

## 八、待确认事项

- 后端在 `adminOnly` 之外是否对所有管理类接口都做了二次角色校验（本次仅逐路由核对，未见统一中间件）。
- `agentos-config` PUT 冲突（409）时的响应结构与前端提示口径。
- 三方 Agent 上传 2GiB 上限的服务端权威判定位置（前端与配置项一致性）。
- Grafana 三只面板（vllm-perf / sglang-perf / log-explore）内部查询语句与管理面指标口径的逐条对应关系。
- `node_service` CLI 中 `check` 子命令未通过 HTTP 暴露，是否需要纳入管理面能力。
