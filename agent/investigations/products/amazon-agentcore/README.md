---
title: "Amazon Bedrock AgentCore 产品调研"
type: investigation
domain: agent
status: active
---

# Amazon Bedrock AgentCore 产品调研

## 结论与研究范围

查阅日期：2026-10-10。依据官方滚动文档和示例仓库，未部署服务或测试生产指标。

AgentCore 是可组合的 Agent 构建与运行平台。Runtime 提供托管执行环境，当前 Harness 则在 Runtime 上提供由 Strands 驱动的托管执行循环。Runtime 的框架中立性不能套用到 Harness 的编排引擎选择上。[B1][B2]

用户提供的 `awslabs/agentcore-samples` 是集成示例仓库，不是 AgentCore 云平台完整源码；样例的开源许可也不意味着可以自行部署相同的 AWS 控制面。[B4]

## 调研入口

- [架构与“七层”辨析](architecture.md)
- [定制和企业托管](customization-and-hosting.md)
- [来源与证据边界](sources.md)

## 与阿里云的主要区别

两者均应按托管执行循环和自带代码分别比较。AWS 更突出可独立采用的服务原语；阿里云文档更突出 Workspace 中的资源组织与 Agent／Team 管理。这是对公开产品界面的分析，不能推导底层实现谁更先进。

当前 AWS Harness 文档列出配置式 Memory、Gateway、Browser 和 Code Interpreter 集成；阿里云托管 Harness 当前支持清单中的长期记忆等限制应单独记录。不能沿用早期“阿里低代码、AWS 只有代码部署”的结论。[B2] 对照 [阿里云模式边界](../alibaba-agentcore/customization-and-hosting.md)。

## 待验证

| 问题 | 实测触发与观察 |
|---|---|
| 隔离 | 双用户恶意交叉访问，分别测试 microVM 与 Instances 设计 |
| 恢复 | 停止／恢复会话，检查文件、循环状态、已执行工具结果 |
| 权限 | 撤销凭据后重复调用，检查缓存、拒绝与审计 |
| 成本 | 固定业务任务，分别统计执行、模型、存储、网关与观测账单 |
| 发布 | 新版本流量切换后检查已有会话行为和回滚 |

## Knowledge Extraction

- [x] 将历史服务清单、架构分析维度和当前能力分开，纳入调研技能。
- [ ] 生产恢复和隔离结论需要部署实验。

[返回产品索引](../README.md)
