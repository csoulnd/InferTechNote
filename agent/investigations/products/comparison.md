---
title: "Agent 产品职责与企业托管对照"
type: investigation
domain: agent
status: active
---

# Agent 产品职责与企业托管对照

## 比较范围

2026-10-10 文档调研结果汇总。各结论的官方来源、源码基线和待验证项保留在对应产品文档。此表比较职责与定制方式，没有统一工作负载测试，不构成性能或价格排名。

| 产品／组件 | 主要定位 | 定制入口 | 运行托管边界 | 研究入口 |
|---|---|---|---|---|
| 阿里云 AgentCore | 构建、运行与治理平台 | Harness 配置、高代码、资产与 Team | 托管实例／Session；本地纳管另算 | [调研](alibaba-agentcore/README.md) |
| Amazon AgentCore | 可组合的 Agent 服务平台 | 托管 Harness、自带应用与服务原语 | Runtime；按计算模式确认隔离 | [调研](amazon-agentcore/README.md) |
| Dify | 应用与工作流开发平台 | 可视化流程、Agent／Build Mode、插件 | Cloud、Community、Enterprise 责任不同 | [调研](dify/README.md) |
| Pi | 可扩展 Harness 与工具库 | 技能、模板、扩展、SDK／RPC | 主要由宿主管理；durable 是独立实验性库 | [调研](pi-agent/README.md) |
| LangChain／LangGraph | Harness 与状态编排库 | 代码、middleware、图 | 库模式由宿主运营 | [调研](langchain/README.md) |
| LangSmith Deployment／Fleet | 相关托管与无代码产品 | 部署配置／无代码 Agent | 随 Cloud、Hybrid、自部署等模式变化 | [边界](langchain/customization-and-hosting.md) |

## 可以提炼的企业平台职责

以下为本批调研的分析框架，不能当作任一厂商官方“七层”。

| 职责 | 需要回答的问题 | 可借鉴的产品方向 |
|---|---|---|
| 产品接入 | 谁能调用 Agent，如何接入业务渠道 | 云平台 Endpoint、Dify 应用 API、Pi RPC |
| 定义与资产 | 定义、工具、技能、模型如何复用与版本化 | Workspace、Registry、应用和插件 |
| Harness 与编排 | 谁控制循环、子任务、流程与审批 | 托管 Harness、Dify Workflow、LangGraph |
| 工具与上下文 | 如何接入模型、知识、记忆与业务动作 | Gateway、RAG、模型接口和工具库 |
| 运行基础设施 | 谁负责隔离、容量、网络和生命周期 | AgentCore Runtime、沙箱、Agent Server |
| 状态与恢复 | 恢复什么，哪些动作会重跑 | 会话历史、文件快照、checkpointer、durable task |
| 治理与运营 | 身份、权限、质量、发布、审计与费用 | 跨层治理服务和运营流程 |

这七项是研究中的责任拆分；数量不代表标准。模型推理基础设施、企业业务系统和知识摄取管道还需要作为外部依赖单独设计。

## 企业托管的关键判断

托管通常具备企业价值，但“企业级”需要具体属性和验收。平台托管进程、托管 Agent 循环、托管状态与托管整个业务事务是不同承诺。

例如用户批准一次外部写入后进程崩溃：历史备份只能提供交互记录；文件快照保存文件；检查点帮助决定从何处恢复；最终写入是否重复仍取决于工具幂等、结果记录和业务系统。平台选型应以这一类场景验证，而不是用“有记忆／有沙箱”替代恢复结论。

## 下一轮可执行验证

选择一个含检索、审批和外部写入的任务，固定模型、工具契约、权限和输入。分别在工具执行前、成功后但结果落盘前、审批等待时中断；记录恢复位置、重复动作、文件丢失和审计关联。再测双租户文件访问与凭据撤销。每个平台先验证可比较的部署模式，能力不对等时明确缺项，避免修改任务掩盖差异。

## Knowledge Extraction

- [x] 完成本批产品责任对照，作为后续研究入口。
- [ ] 经实际验证后，再将稳定结论提炼到 knowledge/agent，当前不把文档分析提升为实测保证。

[返回产品索引](README.md)
