---
title: "Manifest 清单"
type: concept
domain: agent
status: active
---

# Manifest 清单

## 核心问题

Manifest 是什么，它在 Agent、插件或软件制品的发现、校验和加载过程中承担什么职责？

## 一句话解释

Manifest（清单）是由生产者声明制品身份、内容、能力、依赖和兼容约束，供工具在执行制品之前发现与校验的结构化元数据契约。

## 详细解释

Manifest 通常是随软件包、镜像、插件或 Agent 制品一同交付的 JSON、YAML、TOML 等结构化文件。它把加载器需要预先知道的信息从实现代码中分离出来，使系统无需执行未知代码就能回答“这是什么、如何启动、需要什么、能做什么、是否兼容”等问题。

常见声明可分为：

| 类别 | 典型内容 |
| --- | --- |
| 身份 | 名称、版本、制品类型、描述 |
| 入口 | 启动命令、模块入口、交互或 Headless 模式 |
| 能力 | 支持的协议、工具、事件、模型能力 |
| 依赖 | 软件包、运行时、外部服务、加载顺序 |
| 约束 | 平台、架构、宿主版本、权限和配置要求 |
| 完整性 | 内容摘要、签名、许可证和来源 |

Manifest 是声明，不是实现：它可以描述某个 Hook、Tool 或入口，但真正的行为仍由被加载的代码或制品提供。

## 工作原理

```text
发现制品 → 读取 Manifest → 按 schema 校验 → 检查兼容性与策略
        → 解析依赖和入口 → 加载或拒绝 → 记录实际状态
```

1. 生产者按照约定 schema 生成 Manifest，并随制品发布。
2. 消费者先解析和校验字段，再决定是否下载依赖、授予权限或加载代码。
3. 通过校验后，加载器依据入口和依赖声明启动制品；不兼容或缺失的必填项应在执行前失败。
4. 运行时状态、探活结果和动态发现信息不应反写为静态声明，而应由状态接口或锁文件承载。

## 适用边界

- Manifest 只能证明“声明了什么”，不能单独证明实现真实、安全或可用；关键能力仍需验收、签名验证或运行时探测。
- 字段语义取决于具体 schema；同名 `manifest` 文件不代表跨生态兼容。
- 密钥、令牌和环境相关动态值不应写入可分发 Manifest，应通过安全配置注入。
- Manifest 适合稳定、可验证的制品属性，不适合承载频繁变化的运行状态。
- schema 演进应定义版本、必填字段、未知字段处理和向后兼容策略。

## 实践意义

- 在执行第三方 Agent 或插件代码前先读取 Manifest，可提前完成兼容性、权限和供应链策略检查。
- 将“声明校验”和“运行验收”分开：前者验证结构与约束，后者验证声明和真实行为一致。
- 对启动入口、协议、能力与依赖使用机器可读字段，避免加载器依赖目录命名或脚本内容猜测。
- 为 Manifest 定义明确 schema 和版本，并在 CI、发布及安装阶段复用同一套校验器。

## 应用记录

- [第三方 Agent 生态调研](../../../agent/investigations/third-party-agent-ecosystem-research.md)
- [OpenJiuwen 第三方生态调研](../../../agent/investigations/openjiuwen-third-party-ecosystem.md)

## 相关知识

- [Hook 扩展机制](hook-mechanism.md)
- [Cordis 插件运行时](cordis-plugin-runtime.md)
- [Sandbox、OCI 与 Docker 的分层关系](infrastructure/01-sandbox-oci-docker.md)

## 参考资料

- [OCI Image Manifest Specification](https://github.com/opencontainers/image-spec/blob/main/manifest.md)
- [npm package.json](https://docs.npmjs.com/cli/configuring-npm/package-json/)
