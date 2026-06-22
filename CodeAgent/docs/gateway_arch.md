# Gateway 架构设计：SSH 统一入口与 Agent 管理平面

## 概述

本文档描述基于 opencode + jiuwenswarm gateway 构建的 AI Agent 平台架构。核心设计理念是：**一个 Gateway 入口，多端接入，统一身份，体验分级**。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart LR
    subgraph GATEWAY["Gateway"]
        direction LR
        SSH["SSH :22"]
        WEB["Web"]
        IM["IM"]
        ACP["ACP/A2A"]
    end

    SSH --> SSH_EXP["丝滑体验<br/>TUI完整"]
    WEB --> WEB_EXP["基本可用<br/>消息转发"]
    IM --> IM_EXP["轻量交互<br/>消息卡片"]
    ACP --> ACP_EXP["Agent互调<br/>A2A/ACP协议"]

    SSH_EXP & WEB_EXP & IM_EXP & ACP_EXP --> REGISTRY["Agent注册中心<br/>/registry"]

    REGISTRY --> B["容器 B<br/>(Agent)"]
    REGISTRY --> C["容器 C<br/>(Agent)"]

    REGISTRY --> TEAM["Agent Team<br/>Orchestrator<br/>编排层"]
```

## 两条路径，体验分级

### SSH 路径（完整 TUI 体验）

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart LR
    USER["用户"] -- SSH --> GW["Gateway:22"]
    GW -- TCP透传 --> CT["容器:22"]
```

- SSH 字节流端到端透传，Gateway 不做消息解析
- 用户享受原生 SSH 体验：流式输出、tmux 保活、交互式输入
- 底层用 OpenSSH ForceCommand 做路由，不重复造轮子

### 消息路径（非 SSH 渠道）

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart LR
    USER["用户"] -- Web/IM/API --> GW["Gateway"]
    GW -- Message --> HTTP["HTTP/SSH"]
    HTTP --> CT["容器"]
```

- 支持 Web 界面、飞书/DingTalk/Telegram 等 IM 平台
- 一问一答模式，非流式，适合轻量交互
- 在非 SSH 渠道提示用户切换到 SSH 以获取更完整体验

## Gateway 作为管理平面

Gateway 不只是一个消息路由器，更是整个平台的控制平面：

| 功能 | 说明 |
|---|---|
| 用户管理 | 注册、认证、角色 |
| SSH 密钥管理 | 用户公钥上传、自动注入到容器 |
| 容器生命周期 | 按需创建、停止、销毁 |
| 路由配置 | 用户 → 容器的映射自动更新 |

### 首次接入流程

#### 场景一：用户先走 Web

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart TD
    REG["Web 注册"] --> KEY["上传 SSH 公钥/设密码"]
    KEY --> GW["Gateway 创建用户"]
    GW --> CT["Gateway 拉起容器<br/>注入内部 key"]
    CT --> MSG["已就绪，SSH 体验更佳：ssh user@gateway"]
```

#### 场景二：用户先走 SSH

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart TD
    SSH["ssh user@gateway<br/>首次连接，密码登录"] --> CHECK["ForceCommand<br/>检测无容器"]
    CHECK --> LAUNCH["自动拉起容器"]
    LAUNCH --> INJECT["容器创建完成<br/>注入内部 SSH key"]
    INJECT --> TUNNEL["透传进入容器"]
```

#### 场景三：管理员预创建

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart TD
    subgraph ADMIN["管理员操作"]
        A1["Web Admin 创建用户"]
        A2["用户名白名单 AllowUsers"]
        A3["生成随机初始密码"]
        A4["自动拉容器"]
    end

    subgraph USER["用户操作"]
        U1["首次 SSH 登录"]
        U2["输入初始密码"]
        U3["强制修改密码"]
        U4["公钥或新密码免密登录"]
    end

    A1 --> A2 --> A3 --> A4
    A4 --> U1 --> U2 --> U3 --> U4
```

### 密钥分层

Gateway 管理两层密钥：

| 层 | 凭证 | 用户需操作 |
|---|---|---|
| 用户 → Gateway | 公钥 / 密码 | 注册时上传公钥，或首次密码登录 |
| Gateway → 容器 | 内部密钥对（自动生成） | 无感知 |

## Agent 注册中心

每个容器启动时自动注册到 Gateway：

```json
{
  "agent_id": "userA-opencode",
  "host": "container-a",
  "capabilities": {
    "protocols": ["acp", "a2a", "ssh"],
    "models": ["gpt-5.5", "claude-4"],
    "tools": ["write", "bash", "read"],
    "skills": ["python", "react", "docker"]
  },
  "status": "online",
  "owner": "userA"
}
```

### 路由规则

| 来源 | 路由目标 |
|---|---|
| 用户 A 的 SSH 连接 | 容器 A（用户专属 Agent） |
| 用户 A 通过 IM 发消息 | 容器 A |
| Agent A 调用"数据库 Agent" | 注册中心查询 → 容器 B |
| Agent Team 协作 | Orchestrator → 分发给多个 Agent |

## 权限模型

```python
class UserRole(enum):
    GUEST = "guest"      # 自注册，默认受限
    MEMBER = "member"    # 审核通过
    ADMIN = "admin"      # 管理权限

class UserPermissions:
    max_containers: int
    allow_ssh: bool
    resource_limits: ResourceSpec
    can_access_admin: bool
```

## 基础设施

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart TD
    GW["Gateway 机器"]

    subgraph SSH_LAYER["OpenSSHD :22"]
        ALLOW["AllowUsers 白名单"]
        PWD["PasswordAuthentication 首次登录"]
        FC["ForceCommand → route_user.sh"]
    end

    subgraph GW_LAYER["jiuwenswarm gateway.py"]
        CH["WebSocket / IM / ACP 渠道"]
        API["/api/admin/* 管理接口"]
        REG["Agent 注册中心"]
        DOCK["Docker SDK 容器管理"]
    end

    subgraph INFRA["基础设施"]
        RS["route_user.sh<br/>（由 gateway 生成）"]
        DC["Docker<br/>容器运行时"]
    end

    GW --> SSH_LAYER
    GW --> GW_LAYER
    GW --> INFRA
```

## 扩展方向

- **Agent Team**：Orchestrator 编排多个 Agent 协作完成任务
- **A2A Federation**：与外部 Agent 系统互通（Google A2A 协议）
- **专业 Agent**：数据库 Agent、代码审查 Agent、部署 Agent 等，供其他 Agent 调用
- **资源配额**：按用户角色分配 CPU/内存/GPU 限制
