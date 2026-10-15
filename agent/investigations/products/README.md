---
title: "Agent 产品调研索引"
type: moc
domain: agent
status: active
---

# Agent 产品调研索引

基础文档与展示网站分离。此处保留可溯源的产品调研，网站可按产品读取并组织展示。调研主线是产品定位与软件架构：先了解产品面向谁、解决什么核心问题，再理解组件关系、执行路径、实现基础及设计取舍。定制、托管和费用作为辅助内容，未开展部署实测。

每个产品采用 **summary + detail 双文档**：摘要可独立阅读，详细文档集中展开架构、机制、边界与来源。较短的后续研究可用单篇完整报告，不按章节继续拆文件。横向阅读：[产品定位与软件架构对照](comparison.md)。

## 调研顺序

| 顺序 | 产品与核心侧重点（研究判断） | 摘要 | 详细调研 |
|---|---|---|---|
| 1 | 阿里云 AgentCore：统一组织与治理多来源 Agent | [summary](alibaba-agentcore/summary.md) | [detail](alibaba-agentcore/detail.md) |
| 2 | Amazon AgentCore：组合运行基础服务与 Agent 软件 | [summary](amazon-agentcore/summary.md) | [detail](amazon-agentcore/detail.md) |
| 3 | Dify：组合模型、知识与工具，交付业务应用 | [summary](dify/summary.md) | [detail](dify/detail.md) |
| 4 | Pi Agent：可改造、可嵌入的执行 Harness | [summary](pi-agent/summary.md) | [detail](pi-agent/detail.md) |
| 5 | LangChain：用代码组合 Agent 行为与状态编排 | [summary](langchain/summary.md) | [detail](langchain/detail.md) |

初轮证据日期为 2026-10-10；2026-10-12 按新重点调整报告并选择性复核，具体版本与来源日期见各详细文档。

## DSH 专项交流准备

- [与 DSH 团队交流：调研与议题初稿](deepseek-harness/discussion-agenda.md)：保留原生体验的容器托管与行业生态、端到端加速及运行 API。源码基线为 2026-10-09 的固定提交，议题整理于 2026-10-14；仍在讨论收敛中。
- [DSH 穿刺问题清单及进展](../../reports/dsh-integration-issue-log.md)：截至 2026-10-09 的集成验证记录。

## 流程迭代记录

| 已完成产品 | 本次沉淀 |
|---|---|
| 阿里云 AgentCore | 创建 [product-research 技能](../../../.agents/skills/product-research/SKILL.md)，强调部署模式分开、官方架构与研究视图区分、历史恢复与任务恢复分开 |
| Amazon AgentCore | 增补历史七项服务／七维分析／当前模块区分，Harness 与 Runtime 关系，以及计算模式边界 |
| Dify | 增补 Community／Cloud／Enterprise 责任与许可检查；历史 Agent Node 与当前新 Agent 子系统交叉核验 |
| Pi Agent | 增补仓库迁移核验、独立 durable 库的实验性边界、扩展钩子与宿主权限区分 |
| LangChain | 增补框架／Harness／无代码／部署产品区分，内存后端与持久化、节点重执行和遥测位置检查 |
| 2026-10-12 阅读与研究主线调整 | 将五份报告统一为 summary + detail；突出定位、执行路径、实现依据和设计取舍，来源并入详细文档；技能允许较短研究使用单篇完整报告 |

## Knowledge Extraction

- [x] 首次实际调研后沉淀仓库产品调研技能。
- [x] 完成系列后整理跨产品责任比较；部署实验保留为下一轮任务。

[返回 Agent 索引](../../README.md)
