---
title: "与 DSH 团队交流：调研与议题初稿"
type: investigation
domain: agent
status: draft
---

# 与 DSH 团队交流：调研与议题初稿

调研日期：2026-10-14；议题修订：2026-10-15。目标：围绕企业一体机和镜像托管，形成既有探索空间、又有事实抓手的交流议题。部署集成侧重请教软件设计与身份假设；优化加速侧重联合探索与附带信息接口；行业生态侧重以真实需求共建底座。以下“建议、候选、预期成果”均为讨论提案，不代表上游已有承诺。

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

## 三、建议交流的三个议题

以下表述采用“肯定已有能力 → 提出问题 → 寻求合作”的顺序。先对齐设计思路和探索意愿，具体字段与实施细节留待后续专题讨论。

### 1. 保留原生体验的企业级部署集成：对齐 DSH 与容器托管平台的身份假设

**议题背景。** 我们正在将 DSH 以镜像形式部署到企业服务器和一体机，由平台统一分配容器实例、配置资源策略，并通过网关向用户提供服务。部署穿刺已打通主要访问链路，同时也遇到了远程访问下的认证适配、设置能力差异等问题。这使我们意识到，企业集成不仅涉及网络转发，还涉及“谁使用助手、谁管理环境、谁授予权限”的设计假设。我们希望保留 DSH 原生体验，因而需要先向上游请教这些假设如何与平台托管模式对齐，再确定适配方式。[部署穿刺依据](../../../reports/dsh-integration-issue-log.md)

**我们认可的基础。** DSH 将界面、工具、模型和运行能力组织为可组合插件，使我们能够在保留原生体验的同时开展定制。我们已完成一轮容器托管和网关穿刺，希望继续沿用完整 DSH 产品及其演进。

**希望请教的问题。** DSH 当前的软件设计，默认谁是操作者、谁拥有这个运行环境？远程浏览器在什么条件下被视为该操作者？我们理解当前浏览器请求对应同一个操作者 Peer；在企业部署中，则是平台认证企业用户、分配容器实例并管理运行环境，用户通过网关使用自己的助手。这两种身份和信任假设如何对齐，才能在受控部署中保留原生功能，减少因本地访问假设产生的功能差异？

这里需要区分三件事：应用中的操作者身份、请求的 Host/Origin 信任校验，以及容器进程所使用的 OS 账号和权限。三者不能互相替代。我们关注的是平台用户如何被安全地接入其 DSH 实例，而非仅修改请求地址或校验开关。

**希望得到的帮助与合作。** 请 DSH 团队帮助厘清原有设计假设及推荐的托管集成方式。我们可以提供部署现场与问题复现，共同梳理一条保留原生界面和插件体验、由平台承担容器隔离与资源管理的集成路径。先求设计思路一致，再确定必要的上游支持。

依据：当前 Connection 以同一个操作者 Peer 接纳请求，Host/Origin 校验与会话认证分别执行；反向代理已有正式部署支持。[Connection](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/client/connection/README.zh.md#L39)、[公开部署指南](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/user/guide/public-deployments.zh.md#L5)、[穿刺记录](../../../reports/dsh-integration-issue-log.md)

### 2. 从可观测到可优化、可加速：共探 Agent 轨迹优化与推理协同

**议题背景。** 在我们面向工业任务的研究中，优化目标是缩短完成合格任务的时间、降低整体资源成本。一次 Agent 执行的开销同时来自推理、工具等待、重复探索和执行路径选择，因此单独优化模型生成速度并不足够。DSH 的运行记录为分析这些开销提供了基础；我们已有 Hint、工具等待期 KV 缓存保留与语义路由等方向的研究，希望进一步探索如何把观测结果转化为可验证的执行优化，并将运行意图传递给推理侧。相关机制仍需联合验证，尚不能预设它们已经组成可用的自优化系统。[已有研究](../../agent-hint-inference-acceleration-and-nvidia-dynamo.md)、[DSH 接入评估](../../../study-notes/dsh/agent-hint-integration-assessment.md)

**我们认可的基础。** DSH 已有会话事件日志、运行事件、用量/上下文计量和可扩展的遥测机制。这些能力使我们有机会从一次任务的实际执行过程出发分析和改进 Agent，而不仅观察最终结果。

**希望探讨的问题。** 能否将 DSH 的运行轨迹与我们的 Hint 加速、工具等待期的 KV 缓存保留（TTL）策略、语义路由等方向结合，建立“采集轨迹—识别冗余与依赖—生成候选策略—验证收益—反馈更新”的自优化闭环？具体收益和成熟度以联合验证为准。

我们提出的研究判断是：在任务边界明确、重复发生的工业场景中，应逐步将已验证的稳定路径沉淀为更确定、可复用的执行流程，减少无效探索、重复调用和不必要的串行等待；对开放问题和异常情况保留 Agent 的自适应能力。精简以质量、业务约束和必要校验不受损为前提，不能仅追求步骤数量最少。

轨迹优化与推理加速可以相互配合：轨迹帮助识别可合并、可并行或可由确定程序替代的步骤；运行时再向推理侧传递任务阶段、工具等待、上下文复用和路由意图。单次轨迹只是一个已发生路径，需结合数据依赖、外部副作用及多次任务验证后，才能提炼候选执行图/DAG；不能直接将时间先后视为依赖，或把循环和异常分支硬转成固定 DAG。

**希望共建的方向。** Agent 到推理侧的附带信息已有厂商扩展，但跨 Harness、网关和推理后端的共同语义仍值得探索。能否共同定义一组最小运行信号与反馈方式，支持 Hint、缓存保留和路由策略，并用一个真实工业任务验证从轨迹优化到推理加速的闭环？先实现可互通的实验约定，再讨论标准化。

依据与边界：

- DSH 的权威会话日志与事件扩展点提供分析基础，但不等同于已具备全链路 trace 或自动 DAG 优化器。遥测 seam 支持不同捕获后端；随附 OTel 会话后端目前是显式反馈授权后导出，不能假设开启 OTel 就持续采集全部轨迹。[会话日志](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/architecture.zh.md#L124)、[遥测 seam](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/session/session-telemetry/README.zh.md)、[OTel 后端](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/session/session-telemetry-otel/README.zh.md)
- TTL 在这里指工具等待期的缓存保留期限，不是预测工具输出或模型生成超时。Harness 可提供等待和预计复用信号，实际保留策略仍需结合推理侧资源压力；语义路由的对象是模型、能力还是执行路径，也需在试点中明确。[现有 Hint 与推理协同研究](../../agent-hint-inference-acceleration-and-nvidia-dynamo.md)、[DSH Hint 接入评估](../../../study-notes/dsh/agent-hint-integration-assessment.md)
- NVIDIA Dynamo 已有 agent_hints 扩展，不能称该领域完全没有协议；其公开文档仍将更丰富的 TTL/保留控制列为后续方向。这为讨论共同语义提供了参照，不代表已有跨产品标准。[Dynamo 官方说明](https://docs.dynamo.nvidia.com/dynamo/dev/digest/agentic-inference)

### 3. 从开放插件到行业 Agent：以真实需求共建 DSH 行业应用底座

**议题背景。** 我们有行业应用需求，希望以 DSH 的通用运行能力和开放插件机制为基础，组合领域知识、业务工具与执行约束，逐步形成可交付的行业 Agent。现有社区场景探索表明这一方向已有实践，但插件可安装、功能可组合，与行业任务能否稳定完成和验收之间仍有需要验证的环节。当前具体行业样板与验收条件尚待共同选择，因此希望先与 DSH 讨论行业需求如何进入生态、合作双方如何分工，以及交付中遇到的共性问题如何反哺底座。

**我们认可的基础。** DSH 的“一切皆插件”以及 bundle/profile 组合，为行业定制提供了很好的起点。社区已经在工具、界面、模型和场景组合上开展探索，我们希望利用这一开放性，把行业需求与通用底座连接起来。

**希望请教的问题。** DSH 团队如何看待从通用插件生态走向行业 Agent？面对我们的真实行业需求，哪些能力适合沉淀到 DSH，哪些应由行业伙伴以插件、技能和组合包实现？从能组合到能稳定交付，业务知识、行业系统接入、权限约束、结果验收和版本演进可能遇到哪些问题？

**希望开展的合作。** 我们提供具体行业场景、业务约束和验证环境，与 DSH 一起选取样板，将通用能力留在底座，将领域能力沉淀为可复用组合，在原生产品上验证实际业务价值。通过样板识别扩展点和兼容问题，再逐步形成行业交付方法与生态分工。

生态调研结论（2026-10-15）：官方仓库已有插件展示区，并鼓励使用 dsh-plugin 标签发现插件；社区也已有垂直场景尝试，例如独立社区项目 DeepSeek Harness Game Studios。因而不能以“尚无行业 Agent”作为议题前提。更准确的提问是：现有插件与场景探索，如何进一步发展为可验证、可维护的行业方案？本次是公开资料抽样，未完成生态全量盘点，也未验证这些项目的业务效果；未确认官方已有统一的行业方案交付体系。[官方插件展示区](https://github.com/deepseek-ai/deepseek-harness/discussions/categories/show-your-plugins)、[Game Studios 作者介绍](https://github.com/deepseek-ai/deepseek-harness/discussions/9633)、[官方组合包机制](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/docs/user/develop/basic/publish.zh.md)

## 四、推荐的会谈组织

先表达对 DSH 原生体验、可组合架构和运行记录能力的认可，再围绕三件事请教与寻求合作：企业部署的身份假设对齐、轨迹优化与推理协同、真实行业场景共建。各议题先讨论理念和可行方向，技术接口在双方确认意愿后再展开。

建议开场：

> DSH 的原生体验和插件化设计，让我们看到了它作为行业 Agent 底座的潜力。我们已经开展容器托管集成，也在研究如何利用运行轨迹与推理侧信息进一步优化 Agent。希望先请教三个方向：企业托管与 DSH 原有身份假设如何对齐；可观测能力如何进一步连接优化和加速；以及能否结合我们的行业需求，共同探索可交付的行业 Agent。我们愿意提供实际场景、集成问题和验证环境，一起推动这些方向。

第一轮希望形成：部署设计上的共识与求助渠道；一个轨迹优化/推理协同的联合探索切口；一个行业共建样板的选择方向。原先独立的运行 API 议题并入第二项附带信息与协同语义讨论，暂不展开通用任务管理接口清单。

## Knowledge Extraction

- [ ] 交流议题仍在收敛，后续依据确认的接口与验证结果复核可复用知识，本次保留业务背景和原始证据。
