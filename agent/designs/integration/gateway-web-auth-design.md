---
title: "第三方智能体 Web 跳转鉴权方案"
type: design
domain: agent
status: draft
date: 2026-09-10
---

# 第三方智能体 Web 跳转鉴权方案

## 1. 目标与原理

用户从 AgentBox-Manager 管理页面点击 OpenCode 等智能体，通过 Gateway 进入目标实例。Gateway 统一完成身份认证、实例权限检查和 HTTP/WebSocket 代理。

建议入口：`/agents/<agent_type>/<instance_id>/`。路径定位实例，用户身份从可信认证结果获取；不能把 URL 中的 user 当作登录身份。

Cookie 存在用户电脑的浏览器里，但归属于设置它的访问主机。浏览器请求目标地址时，自动把符合 Domain/Path/Secure/SameSite 等规则的 Cookie 放入请求头；远程 Gateway 从请求读取，不需要访问用户电脑。HttpOnly 禁止页面 JS 读取，不阻止浏览器发送。

假设管理端为 `http://192.168.1.10:8090`：

| Gateway 浏览器入口 | 是否跨域 | 当前 IAM Cookie |
|---|---|---|
| `http://192.168.1.10:8080` | 是，端口不同 | 可以携带，Cookie 不按端口隔离 |
| `http://192.168.1.20:8080` | 是，IP 不同 | 不会携带 |
| `http://gateway.example.com` | 是，主机名不同 | 不会携带 |

“同主机”指浏览器 URL 中的主机名/IP，不代表进程必须部署在同一物理服务器。普通链接跳转不能自定义 Authorization Header；配置 CORS 也不会把一个主机的 Cookie 变成另一个主机的凭证。

## 2. 仓库现状

- 管理端默认暴露 `8090`，Nginx 将 `/api/` 代理到内部后端；当前管理端配置未发现智能体 Gateway 代理入口，未获得实际部署的 Gateway 地址。
- 登录接口设置 HttpOnly Cookie：`Path=/`、`SameSite=Lax`，未指定 Domain；Secure 根据请求是否 HTTPS 设置。
- 普通登录 JWT 包含 `sub`（用户 UUID）、`username`、`role`、`type=access`、`jti`、`iat`、`exp`；认证依赖优先读取 Bearer Header，再读取 Cookie。
- `/api/v1/auth/verify` 校验普通登录 JWT 并进行角色资源权限判断，不应直接等同于实例归属校验；普通 JWT 验证本身不查用户最新状态。
- 已有 OAuth2 授权码、换 Token 和 userinfo 接口。OAuth2 JWT 为 `type=oauth2_access`，不含 role；userinfo 校验 Token 后查询用户是否存在且启用。
- OAuth2 换码实现当前传入空 username 签发 Token，因此应以 `sub` 和 userinfo 返回的身份为准，不依赖 JWT 中 username 一定有值。

## 3. 四种方案

### 方案一：同主机共享 IAM Cookie

**条件：** 管理端与 Gateway 的浏览器入口使用相同 IP/主机名，允许端口不同，且满足 Cookie 的协议等限制。

```text
管理页面点击实例 → 浏览器携带 IAM Cookie 请求 Gateway
→ Gateway 校验 JWT → 检查实例权限 → 代理智能体
```

**实现：** 管理端拼接实例链接；Gateway 读取配置的 IAM Cookie，通过 IAM 校验接口验证身份，再查询实例访问权限。也可设计本地验签，但必须建立可信密钥配置，不能把签名密钥暴露给浏览器。

**会话：** 复用 IAM Token 生命周期；过期时返回管理端重新认证并恢复目标实例。不要假定页面跳转会自动调用刷新接口。

**区别：** 改动少，无额外登录；同主机各端口共享 Cookie 范围，必须属于相同信任边界。Gateway 收到 Cookie 后仍需主动验证，不能只判断 Cookie 存在。

### 方案二：管理端同域反向代理 Gateway

**条件：** 管理端 Nginx 能访问远程 Gateway，智能体 Web 服务支持或能适配代理路径。

```text
浏览器访问 manager.example.com/agents/<type>/<instance_id>/
→ 管理端入口校验 IAM Cookie 和实例权限
→ 内部 Gateway → 智能体实例
```

**实现：** Nginx 新增 `/agents/` 路由和鉴权子请求，转发到 Gateway；适配静态资源、重定向、API、WebSocket 和长连接。Gateway 可以继续验 Token；如果改为信任入口注入的身份 Header，必须限制入口来源并覆盖客户端同名 Header。

**会话：** 使用管理端现有 Cookie，过期后回到管理端认证。

**区别：** 浏览器视角完全同源，Gateway 物理部署位置不受限；主要成本在代理和子路径兼容。仓库 Grafana 的 auth_request 模式可参考，但现有 proxy-verify 仅允许管理员，不能直接用于普通用户实例鉴权。

### 方案三：Gateway 二次登录并建立独立会话

**条件：** 任意域名/IP；允许用户首次进入 Gateway 再输入一次账号密码。

```text
点击实例 → Gateway 无会话 → Gateway 登录页
→ Gateway 后端调用 IAM 登录接口 → 建立 Gateway 会话
→ 检查实例权限 → 返回目标实例
```

**实现：** 新增 Gateway 登录页、登录接口和服务端会话存储；继续使用 IAM 账号，无须新增用户库。IAM Token 保存在 Gateway 服务端，浏览器只持有 Gateway 的随机会话 Cookie。后端调用 IAM 登录不会自动在浏览器建立 Gateway Cookie，必须显式设置。

**会话：** Gateway 负责刷新、过期及退出；会话不得因 IAM Token 过期而无限延长。平台退出与 Gateway 退出默认不联动，若要求统一退出需新增会话吊销机制。

**区别：** 不依赖跨主机 Cookie；新增独立会话功能且 Gateway 会处理用户密码，交互成本高于单点登录。登录链路采用 HTTPS。

### 方案四：Gateway 接入现有 OAuth2 授权码流程

**条件：** 任意域名/IP；Gateway 作为有后端的 OAuth2 客户端接入 IAM。

```text
点击实例 → Gateway 无会话
→ IAM GET /api/v1/oauth2/authorize
→ 用户登录/确认授权 → Gateway 回调收到 code 和 state
→ Gateway POST /api/v1/oauth2/token 换 Token
→ Gateway GET /api/v1/oauth2/userinfo 获取身份
→ 建立 Gateway 会话 → 检查实例权限 → 返回目标实例
```

**实现：** Gateway 新增发起授权、回调、换码和会话模块。随机 state 与浏览器登录事务绑定，回调严格校验；目标实例保存在服务端事务中，回跳地址限制为允许的本地路径。client_secret 和 OAuth2 Token 只在后端使用。

**仓库适配：**

- 当前 OAuth2Service 按单个 `OAUTH2_CLIENT_ID/SECRET/REDIRECT_URI` 校验；若保留 SkillHub 并新增 Gateway，应扩展客户端注册及各自回调白名单。
- userinfo 仅接受 OAuth2 Token，不接受普通登录 JWT，返回 `id/username/login/name`；实例授权需要独立实现。
- 当前换码流程不提供 OAuth2 refresh_token；会话过期重新发起授权，不要假定支持刷新授权。
- 用户已有 IAM 会话时可以复用身份，是否需要点击授权取决于授权页逻辑；并非天然静默登录。

**会话：** Gateway 保存独立 Cookie 和服务端会话，明确 Token 到期、用户停用复查及退出策略。现有 userinfo 会查用户启用状态，但 Gateway 仅登录时调用一次不会自动感知后续停用。

**区别：** 用户密码只交给 IAM，适合独立 Gateway 的长期接入；比二次登录多了授权事务和客户端配置。仓库实现是 OAuth2 加自定义 userinfo，不能直接视为完整 OIDC 服务。

## 4. 对比与选型

| 维度 | ① 共享 Cookie | ② 同域代理 | ③ 二次登录 | ④ OAuth2 |
|---|---|---|---|---|
| 不同浏览器主机地址 | 不支持当前 Cookie 直接共享 | 通过统一入口解决 | 支持 | 支持 |
| 用户操作 | 直接进入 | 直接进入 | 首次再次输入密码 | 跳转 IAM，可能确认授权 |
| Gateway 凭证 | IAM Cookie/JWT | IAM JWT 或可信代理身份 | 自己的会话 Cookie | 自己的会话 Cookie |
| 主要改动 | Gateway 验证和权限 | Nginx、鉴权、路径兼容 | 登录 UI、会话管理 | OAuth2 客户端、多客户端配置、会话 |
| 相对成本 | 低 | 中，取决于 Web 兼容 | 中 | 中到高 |

选型建议：同 IP 仅端口不同选①；能统一入口且支持子路径选②；允许重复登录选③；独立域名/IP 的长期集成优先④。实际入口地址仍需部署环境确认。

## 5. 共同实现边界与验收

- 鉴权覆盖页面、API 和 WebSocket 握手；长连接明确会话过期后的断开或复核策略。
- 根据可信用户 ID 检查实例归属/共享权限，修改 URL 不能访问其他用户实例。
- 智能体端口不向用户直接开放，防止绕过 Gateway；禁止客户端指定任意上游代理地址。
- JWT 不放在实例 URL；Cookie 会话配套写操作 CSRF 防护和 WebSocket Origin 校验，生产环境使用 HTTPS。
- 不把平台 IAM Cookie、Gateway 会话 Cookie、刷新 Token 或 OAuth2 密钥原样转发给第三方智能体；只传递业务确需的可信身份信息。
- 验收覆盖：已登录进入、未登录恢复目标、越权拒绝、Token 到期、用户停用策略、退出、静态资源/API/WebSocket，以及上游端口绕过访问。

## 6. 依据与关联

代码依据：AgentBox-Manager 工作区，核对日期 2026-09-10。以下路径均相对该代码仓库：

| 文件 | 依据 |
|---|---|
| `deploy/docker-compose.yml`、`deploy/nginx.conf` | 管理端入口和代理 |
| `backend/app/api/v1/auth.py` | Cookie 设置、登录、校验接口 |
| `backend/app/iam/tokens.py`、`backend/app/iam/security.py` | JWT 字段、认证与 userinfo 用户状态检查 |
| `backend/app/api/v1/oauth2.py`、`backend/app/services/oauth_service.py` | 授权码接口、单客户端校验与实际换码行为 |

关联文档：[AgentOS Web 直通方案](design-agentos-web-direct.md)。本文补充浏览器跳转的身份交接设计；四种方案为待选设计，不表示 Gateway 已实现。
