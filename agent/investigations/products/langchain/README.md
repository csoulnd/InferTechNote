---
title: "LangChain 产品与生态调研"
type: investigation
domain: agent
status: active
---

# LangChain 产品与生态调研

## 范围与结论

查阅日期：2026-10-10。研究当前 Python 官方文档和关联商业平台，不安装 SDK、不测试部署，也不把本次结论推广到所有 JS 版本。文档滚动更新，本次没有绑定源码提交。

LangChain 当前提供 `create_agent`，通过模型、工具、提示词和 middleware 组合可定制 Harness，Agent 底层建立在 LangGraph 上。[L1] LangGraph 是较低层的状态编排框架／运行库，适合将确定性步骤和模型决策组合。[L2]

LangSmith 是相关平台，负责观测、评估及部署。它的商业托管能力不能写成安装 LangChain 包后默认获得的能力。[L5] Deep Agents 提供更完整的 Harness，Fleet 提供无代码构建；它们应作为关联产品分别辨认，而不是叫作 LangChain 的几个内置开关。[L6][L7]

## 与本系列其他产品的关系

LangChain／LangGraph 可作为 AgentCore Runtime 中应用的执行逻辑，二者关注的职责能够组合。LangSmith Deployment 则在托管运行方面形成另一种平台选择。比较时分别比较 Harness、编排、治理与部署，不能把整个生态与单个框架包作排行榜。

- [架构与职责](architecture.md)
- [定制、恢复及企业托管](customization-and-hosting.md)
- [来源登记](sources.md)

## 待验证

针对有审批和外部写入的任务，验证断电后恢复节点、是否重复执行写入、审批身份是否与 thread 正确绑定。对托管方案验证并发、升级兼容性、数据位置与权限撤销。当前文档能力不构成已验证 SLA 或性能结论。

## Knowledge Extraction

- [x] 将框架、Harness、运行库和商业部署的生态边界纳入技能。
- [x] 将 checkpoint／store／外部效果分开分析纳入技能。
- [ ] 发布与恢复行为待运行验证。

[返回产品索引](../README.md)
