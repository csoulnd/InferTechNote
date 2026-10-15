---
title: "与 DSH 团队交流：调研与议题初稿"
type: investigation
domain: agent
status: draft
---

# 与 DSH 团队交流：调研与议题初稿

调研日期：2026-10-14；议题修订：2026-10-15。目标：围绕企业一体机和镜像托管，形成既有探索空间、又有事实抓手的交流议题。行业生态侧重合作方案；部署与 API 侧重上游接口/特性支持；性能侧重共同定义方向与联合试点。以下“建议、候选、预期成果”均为讨论提案，不代表上游已有承诺。

## 一、先建立共同认识：DSH 的软件理念

DSH 是可组合的 Agent 运行框架。它的“一切皆插件”涵盖模型适配器、工具、会话日志、沙箱乃至 agent loop，扩展主要通过插件、事件和配置完成。它的价值在于让不同执行能力与策略组合成不同产品。[README](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/README.zh.md#L5)、[架构](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/architecture.zh.md#L9)

原生交付单位是 bundle 与 profile：bundle 分发配置及挂载代码，profile 保存有序组合、外部插件和用户覆盖。文件系统和进程提供方可以共同切换执行环境，减少为远程沙箱维护专用 fork 的需要。[组合机制](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/architecture.zh.md#L17)、[能力替换](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/architecture.zh.md#L138)

会话日志强调“模型可见即已记录”，为上下文重建、续跑与观测提供基础；这不等同于所有外部副作用都能重放，也不等同于完整企业审计。[会话日志](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/architecture.zh.md#L128)

重要的成熟度前提：DSH 是开发者预览，接口可能破坏兼容；安全说明明确尚未接受安全审计，不能把它当成不可信负载唯一的隔离措施。企业化应围绕外部隔离、治理和受支持的扩展接口设计。[预览状态](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/README.zh.md#L13)、[安全边界](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/SAFETY.zh.md#L7)

**行业生态方向的核心张力：DSH 已支持深度定制组合，但如何保留原生界面、插件生态和持续升级能力，将完整产品从本地使用形态拓展为企业托管形态？**我们希望行业伙伴能够继续组合和定制 DSH，由服务器承载运行时、企业平台管理入口与实例，用户获得完整的原生体验。这是本轮希望共同探索的产品方向，尚不是 DSH 的产品路线声明。

## 二、企业托管已经走到哪里

目标场景：DSH 镜像运行于服务器；平台统一管理实例和资源策略；浏览器通过 Gateway 访问各实例；配置、凭据、文件和升级由企业控制面承接。同时保留 DSH 原生前端及其插件组合体验，避免为了托管而只取核心或 SDK、另行重建整套 App。

穿刺清单截至 2026-10-09 记录 16 项独立根因：13 项验证完成、2 项 Manager 待开发、1 项文件服务待接入。“完成”包括兼容方案和接受产品边界，不等于全部原生根治。[穿刺清单](../../../reports/dsh-integration-issue-log.md)

| 类别 | 现有事实 | 对议题的启发 |
|---|---|---|
| 网关透明转发 | Host、Cookie、WS 路由、业务 query、multipart boundary 已完成验证 | 沉淀代理一致性规范与测试；这些是平台转发问题 |
| 兼容未根治 | /plugins/??... 特殊 query 通过 DSH 插件编码到路径规避 | Gateway 应原生保留请求语义，减少长期兼容负担 |
| 远程管理边界 | 已撤销伪装 loopback 与客户端补丁，模型/凭据通过受控配置注入 | 探讨受控管理能力，避免依赖本机桌面入口 |
| 版本与持久目录 | 镜像 revision 触发 profile 刷新，保留 sessions、storages、凭据 | 还需要组合版本、配置迁移与会话兼容的共同约定 |
| 平台待建设 | 自助配置编辑、灰度升级与回滚缺失 | 主要由 Manager 承担，需要 DSH 提供必要状态与生命周期接口 |
| 文件交付 | 预览正常，下载待统一文件服务接入 | 平台负责存储/权限/生命周期，DSH 提供交付物集成点 |

需要区分远程访问与完整的企业托管产品形态。当前仓库已有 public-url、trusted-host、路径前缀、Cookie 作用域及 TLS 的正式反代说明，支持服务器运行、浏览器经代理访问；但这些支持本身不能证明原生前端可以独立交付、绑定不同远程实例，并在受控授权下完整使用产品能力。[部署指南](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/user/guide/public-deployments.zh.md#L5)

默认 Web 形态由 Host 提供已构建前端，入口 HTML 经运行时渲染注入启动信息与插件资源，并处理入口认证；因此单独搬走静态目录，不能直接得到同等功能的独立前端。前端由主进程提供本身不阻止远程访问，但前端交付、实例连接、插件资源装配和认证之间的关联，需要为托管场景共同定义。[前端交付机制](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/host/frontend-static/README.zh.md#L44)

非 loopback 访问时设置能力受限，是我们已经确认的产品差异。待讨论的是：保留原生 UI 时，哪些本地能力应继续限定本机，哪些可通过明确授权和能力声明提供给远程用户。前后端独立部署是候选路线之一，具体形式仍需与 DSH 确认。

也不能直接把一个 DSH 实例视为共享多租户服务：当前 Connection 中所有接纳请求代表同一个操作者 Peer，启动 token 在根路径换 Cookie，HTTP 不接受 Authorization token；已有 admit 和 connection/request 扩展点，但其存在不等于企业 IAM 已实现。试点可先明确一用户一实例或其他隔离域。[认证与准入](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/client/connection/README.zh.md#L39)

资源策略需拆成三个层次：实例 CPU/内存/存储与并发；模型路由、请求和 token 预算；工具、文件、网络与凭据访问。它们分别涉及平台调度、Agent 运行时和执行隔离，不能只用一份启动 YAML 代替治理。

版本注意：清单主要验证 0.2.0-rc.2；本地仓库 HEAD 为 d743267388（2026-10-09，合入 release-0.2.1-alpha.2）。新文档能力是否能用于当前镜像，需要另做版本对应与验证；本次未重新部署或运行功能测试。

## 三、建议提交的三个议题

### 1. 面向千行百 Agent 的容器托管：如何保留 DSH 原生体验，支撑企业一体机与行业定制？

**核心问题：如何将完整 DSH 及其行业定制组合部署在独立容器实例中，由平台统一接入、分配资源和管理生命周期，同时保留原生界面、插件生态与持续升级能力？**

以平台托管 DSH 容器实例为主线：平台把用户或明确的隔离域映射到独立实例，DSH 在实例内提供原生助手与插件组合能力。实例内可以包含多个 Session；Session 划分和 Peer 身份本身不构成文件、凭据与进程资源的隔离。具体容器、挂载、网络和执行权限由平台配置和验证。

bundle/profile 已提供深度定制与组合机制。希望与 DSH 共同探索的是：行业伙伴如何将这些组合连同原生产品一起制作成可托管镜像，在企业服务器或一体机上交付和运营。

建议的职责分工（待双方确认）：

| 事项 | 平台侧职责 | 希望 DSH 提供的集成支持 |
|---|---|---|
| 用户与实例接入 | IAM、用户/隔离域到实例的映射、网关路由与访问控制 | 受支持的入口认证适配、远程连接和原生 UI 功能授权方式 |
| 实例资源与执行隔离 | 容器调度、CPU/内存配额、存储挂载、网络与凭据范围 | 可配置的运行参数、必要状态与用量输出，遵循实例内执行约束 |
| 原生前端与插件 | 托管访问入口、转发 HTTP/WS 与资源请求 | 明确启动信息、插件资源装配和版本匹配机制；独立前端交付按实际需要探索 |
| 配置与行业组合 | 构建并验证镜像、注入平台配置和凭据、区分用户可修改范围 | 支持配置注入与校验，明确原生设置及插件操作在远程场景的行为 |
| 生命周期与用户数据 | 启停、持久卷、升级灰度、回滚和文件服务 | 就绪/版本信息、优雅停机，以及配置和会话数据的兼容与迁移说明 |

讨论内容：

- 完整 DSH 容器经网关访问时，哪些原生功能仍受本地地址判断约束？哪些可在明确授权下开放给远程用户？
- 保留 Host 提供前端的现有方式是否足够？何时才需要独立前端，以及如何支持远程实例连接、启动信息和客户端插件装配？前后端分离不作为容器托管的先决条件。
- 平台已有身份认证与实例隔离时，DSH 需要哪些最小适配接口，才能减少 Cookie 交换、客户端补丁和版本升级中的定制负担？
- 配置、插件和凭据哪些随镜像发布，哪些由平台注入，哪些允许用户维护？如何避免持久目录让新镜像继续加载旧组合？
- 行业伙伴、平台与一体机厂商如何围绕同一套 DSH 原生产品分工，维护镜像、插件组合、部署验证与兼容责任？

事实抓手：DSH 已有 bundle/profile、反向代理部署与请求准入扩展点；默认 Web 前端由 Host 提供并动态装配，当前浏览器请求代表同一个操作者 Peer；穿刺中仍存在远程功能差异及兼容适配。这些是容器托管集成的讨论依据。[组合机制](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/architecture.zh.md#L17)、[前端交付](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/host/frontend-static/README.zh.md#L44)、[认证与准入](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/client/connection/README.zh.md#L39)、[穿刺记录](../../../reports/dsh-integration-issue-log.md)

**侧重点：共定义容器托管参考方案与行业生态分工，推动必要的上游集成支持。**

预期成果：一套保留原生体验的容器托管参考部署、一份平台与 DSH 的最小集成清单，以及一个行业定制镜像的部署/升级验证样板。当前路线由平台承担实例隔离和资源管理；单个 DSH 进程原生服务多个租户，作为独立的远期方向，不纳入本议题的建设前提。

### 2. 从模型快到任务快：共建端到端 Agent 加速基准

**核心问题：一体机上的加速，最终能否降低完成一次合格任务的时间与成本？**

DSH 已有 Harness 性能基准，但部分续跑/工具基准不包含真实模型网络调用，不能据此推算企业端到端吞吐。[基准范围](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/benchmarks/agent-continuation/README.md#L27)

建议共同测量：

- 业务层：任务成功率、完成时间 p50/p95、每次成功成本。
- 运行层：排队、模型往返、工具/PTC 启动、压缩、子 Agent 和文件传输耗时。
- 推理层：首 token、生成速率、缓存读写、并发吞吐和资源使用。

候选路线：缓存友好的上下文构造、PTC/批量工具执行、合理子 Agent 并行、模型路由、实例预热。固定质量门槛，测冷/热启动与并发，记录重试和失败；先测基线，再设收益目标。

外部案例只说明可能性：Anthropic 自身复杂研究测试中 PTC 平均 token 降低约 37%；vLLM 前缀缓存主要节省 prefill，不能视为 decode 加速。这些收益不能直接套用到 DSH。[PTC 案例](https://www.anthropic.com/engineering/advanced-tool-use)、[vLLM 缓存说明](https://docs.vllm.ai/en/v0.22.1/features/automatic_prefix_caching/)

**侧重点：共定义基准与联合试点。**
预期成果：可重复的两类负载（多工具分析、长会话代码任务）和对照报告。可能的上游支持：关联模型、Agent、工具、网关的 trace/usage 导出及能力声明。

### 3. 从可调用到可运营：探索企业 Agent 运行 API 与能力协商

**核心问题：平台如何可靠地提交、观察、取消和接续 Agent 工作，并据此调度与计量？**

事实抓手：SDK 目前通过 stdio JSON-RPC 驱动运行时；文档明确无协议版本协商、无取消/会话关闭方法。messageId 表示排队回执，并非任务结果标识；idle 也不能直接解释为某条输入的独立终态。[SDK 语义](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/sdk/protocol/README.zh.md#L57)、[已知限制](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/sdk/protocol/README.zh.md#L117)

讨论内容：

- 区分 Session、Message、Run/Task，保留 inbox/steering 语义，同时增加平台可追踪的执行标识。
- 定义提交、状态查询、事件订阅、取消、释放和终态；取消一个任务是否能避免关闭整个实例？
- 定义断连续读、重复事件、幂等提交和结果未知的处理方式。
- 握手是否公开版本与 capabilities：模型、模态、工具、缓存、预算、取消、恢复、用量？
- 如何将任务、后代 Agent 与耗时/token/交付物关联起来？

**侧重点：推动上游支持，逐步探索跨产品约定。**
预期成果：一个最小运行接口草案和断连/取消/多实例验证样板。先定义可实施语义，再讨论标准化；不预设一定采用 REST，也不把浏览器 Remote 协议直接当成稳定企业 SDK。

## 四、推荐的会谈组织

对外围绕三个议题展开：容器托管与行业生态、端到端加速、运行 API。

容器托管议题统一讨论原生产品体验、平台与运行时分工、行业镜像交付及最小集成支持。资源与隔离以平台管理独立实例为主，避免把讨论扩展成 DSH 原生多租户企业版本的建设。

建议开场：

> 我们已经把 DSH 放进镜像托管、多实例和网关转发场景，完成了一轮端到端穿刺。我们希望保留 DSH 原生界面和插件组合能力，把定制后的完整产品以容器实例运行在企业服务器或一体机上，由平台负责实例隔离和资源管理，并持续继承上游演进。接下来希望共同讨论适合这种场景的托管产品形态、平台治理接口，以及端到端任务加速。我们可以提供部署现场、问题复现和验证样板，与 DSH 一起确定支持范围和演进路径。

第一轮希望形成三类结果：明确合作范围；确定少量上游接口诉求的优先级；选出一个行业样板与一个性能验证负载。议题标题现在即可使用，具体行业、硬件、业务数据和验收门槛待补充后收敛。

## Knowledge Extraction

- [ ] 交流议题仍在收敛，后续依据确认的接口与验证结果复核可复用知识，本次保留业务背景和原始证据。
