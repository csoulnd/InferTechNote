---
title: "Agent 产品调研索引"
type: moc
domain: agent
status: active
---

# Agent 产品调研索引

基础文档与展示网站分离。此处保留可溯源的产品调研，网站可按产品读取并组织展示。当前批次关注定义、架构、定制和企业托管，未开展部署实测。

## 调研顺序

| 顺序 | 产品 | 状态 |
|---|---|---|
| 1 | [阿里云 AgentCore](alibaba-agentcore/README.md) | 文档调研完成 |
| 2 | [Amazon Bedrock AgentCore](amazon-agentcore/README.md) | 文档调研完成 |
| 3 | [Dify](dify/README.md) | 文档与部署结构调研完成 |
| 4 | [Pi Agent](pi-agent/README.md) | 文档与包结构调研完成 |
| 5 | LangChain | 待研究 |

## 流程迭代记录

| 已完成产品 | 本次沉淀 |
|---|---|
| 阿里云 AgentCore | 创建 [product-research 技能](../../../.agents/skills/product-research/SKILL.md)，强调部署模式分开、官方架构与研究视图区分、历史恢复与任务恢复分开 |
| Amazon AgentCore | 增补历史七项服务／七维分析／当前模块区分，Harness 与 Runtime 关系，以及计算模式边界 |
| Dify | 增补 Community／Cloud／Enterprise 责任与许可检查；历史 Agent Node 与当前新 Agent 子系统交叉核验 |
| Pi Agent | 增补仓库迁移核验、独立 durable 库的实验性边界、扩展钩子与宿主权限区分 |

## Knowledge Extraction

- [x] 首次实际调研后沉淀仓库产品调研技能。
- [ ] 完成系列后提炼跨产品责任比较。

[返回 Agent 索引](../../README.md)
