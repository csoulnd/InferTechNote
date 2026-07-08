# 鉴权安全：OAuth / API Key / Agent Rail

> 介绍级文档：建立原理认知与场景映射，深入实践见参考链接。

## 原理

**API Key** 适合简单机器间鉴权，但静态 Key 易泄露、难轮换。**OAuth 2.0**（RFC 6749）的 **Client Credentials** 模式适合服务间调用：客户端用 client_id/secret 换短期 access_token，vLLM 网关或 Gateway 校验 token 而非裸 Key。

**Agent Rail（护栏）** 在 Agent 执行链路中拦截风险：输入过滤（prompt 注入）、执行门控（危险 bash）、输出审查。NVIDIA **NeMo Guardrails** 是行业参考实现；OpenCode 内置 permission 系统；Claude Code 有 deny-first 权限门控。

与 **容器沙箱**（namespace、seccomp、Landlock）形成纵深：Rail 管 Agent 行为，沙箱管 OS 边界。

## 典型应用场景

- 平台 OAuth 统一鉴权，替换各租户静态 vLLM API Key
- Gateway 入口校验用户身份与配额，再路由到 Agent 容器
- NeMo Guardrails 拦截恶意 prompt 或敏感数据外泄
- CC Permission gating 与 jiuwenbox policy 对照设计平台护栏策略

## 参考链接

### 官方标准

- [RFC 6749 OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc6749)
- [Auth0 Client Credentials](https://auth0.com/docs/secure/tokens/client-credentials-tokens)
- [OWASP API Security Top 10](https://owasp.org/API-Security/)

### 开源项目

- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [OpenCode](https://github.com/opencode-ai/opencode) — 权限 / Rail 参考

### agentos 对照

- `agentos/jiuwenswarm/jiuwenbox/` — policy YAML（namespace、syscall、Landlock）

### 视频

- B 站搜索：`大模型 Agent 安全防护 Rail 护栏实战`
