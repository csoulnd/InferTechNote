---
title: "Dify 产品调研"
type: investigation
domain: agent
status: active
---

# Dify 产品调研

## 结论与范围

查阅日期：2026-10-10。研究官方产品说明、部署清单及仓库内技术文档，未部署。源码基线为 `0afbe55884d62cb492e4cdbbf470954f252fb474`，由 main 分支远程引用查询取得，不等同于所有云端实例版本。

Dify 是 LLM 应用开发平台，围绕可视化工作流、知识处理、模型与工具集成组织应用。[D1] 当前调研必须包含新的 Dify Agent：2026-08-27 官方说明其支持对话构建、技能与 CLI 工作环境，不应只依据 2025 年 Agent Node 的介绍描述现状。[D4][D5]

最值得借鉴的设计是把业务流程与自主任务分开：Workflow 约束步骤之间的顺序和边界，Agent 在某一步内部选择动作。这种组合比“一个提示词就是整个应用”提供更明确的可维护结构。

## 核心能力与入口

| 面向对象 | 定制方式 | 责任 |
|---|---|---|
| 应用作者 | 可视化图、模型、提示词、知识检索、工具 | 业务输入输出和流程边界 |
| Agent 作者 | 新 Agent 的 Build Mode、指令、技能、工具 | 任务目标、执行权限和验收 |
| 插件开发者 | 工具、模型或 Agent Strategy 扩展 | 新能力的代码和契约 |
| 平台运营者 | Cloud、Community 或 Enterprise 部署选择 | 版本、容量、租户与治理 |

公开仓库当前 Dify Agent 技术文档指向 Agenton 组合的 Pydantic AI 执行与 FastAPI 服务；这是该子系统依据，不能概括成“整个 Dify 基于 Pydantic AI”。[D6]

- [架构分析](architecture.md)
- [定制与托管](customization-and-hosting.md)
- [来源登记](sources.md)

## 待验证

用同一个审批加检索任务分别验证 Workflow、Agent Node 与新 Agent 的流程导出、异常重试和日志关联。自部署时测试断电恢复、插件故障、并发和不同用户的文件访问。Enterprise 的隔离与权限需要对应版本和合同验证。

## Knowledge Extraction

- [x] 将社区、Cloud、Enterprise 的能力及运维责任区分纳入技能。
- [x] 将历史介绍与当前新增子系统的交叉核验纳入技能。
- [ ] 流程恢复、权限与隔离待实测。

[返回产品索引](../README.md)
