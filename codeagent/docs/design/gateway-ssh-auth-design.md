# Gateway SSH Public-Key Authentication Design

## 概述

本文档描述了 Gateway 在 cc-switch 场景下通过 SSH 进行终端交互的完整认证链路设计，涵盖 SSH 密钥认证原理、密钥管理方案对比、TUI 载体分析，以及最终选型方案 A（端到端公钥认证）的详细实现。

**设计目标**：用户通过 Gateway 拉起 cc 容器实例后，通过本地终端模拟器建立 SSH 连接，实现端到端加密的 TUI 交互。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px', 'fontFamily': 'Segoe UI'}}}%%
flowchart LR
    Client["🖥️ 客户端<br/>浏览器 + 本地终端"] ===|"WS 控制面"| Gateway["🌐 Gateway<br/>认证 / 路由 / 透传"]
    Gateway ===|"SSH 数据面"| Instance["📦 容器实例<br/>Claude Code + sshd"]
```

---

## 一、SSH 公钥认证基础原理

### 1.1 密钥文件分布全景

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
block-beta
    columns 2
    
    block:Client:2
        columns 1
        C1["~/.ssh/id_ed25519<br/>🔒 用户私钥 (600)"]
        C2["~/.ssh/id_ed25519.pub<br/>📤 用户公钥 (644)"]
        C3["~/.ssh/known_hosts<br/>📥 主机公钥缓存"]
    end
    
    block:Server:2
        columns 1
        S1["~/.ssh/authorized_keys<br/>📥 客户端公钥列表 (600)"]
        S2["/etc/ssh/ssh_host_*_key<br/>🔒 主机私钥"]
        S3["/etc/ssh/ssh_host_*_key.pub<br/>📤 主机公钥"]
    end
    
    C1 -->|"永不离开"| C1
    S2 -->|"永不离开"| S2
    C2 -.->|"部署到"| S1
    S3 -.->|"首次连接存入"| C3

    classDef keyNode fill:#fef2f2,stroke:#dc2626,color:#7f1d1d
    classDef pubNode fill:#f0fdf4,stroke:#16a34a,color:#14532d
    classDef cacheNode fill:#fefce8,stroke:#ca8a04,color:#713f12
    
    class C1,S2 keyNode
    class C2,S3 pubNode
    class C3,S1 cacheNode
```

### 1.2 完整认证握手流程（四阶段）

整个 SSH 连接涉及**两次独立的非对称加密验证**：第一次用主机密钥验证服务器身份（防中间人），第二次用用户密钥验证用户身份（替代密码）。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '12px', 'sequenceNumberColour': '#0369a1'}}}%%
sequenceDiagram
    autonumber
    participant Client as 🖥️ 客户端
    participant Server as 🖧 服务端

    rect rgb(219, 234, 254)
        Note over Client,Server: 🔵 阶段一：TCP + 版本协商
        Client->>Server: TCP 三次握手
        Client->>Server: SSH 版本协商 (SSH-2.0-OpenSSH_9.x)
    end

    rect rgb(255, 237, 213)
        Note over Client,Server: 🟠 阶段二：密钥交换 (ECDH)
        Client->>Server: 临时公钥 EC
        Server-->>Client: 临时公钥 ES + 主机公钥 HP
        Note over Client,Server: 双方 ECDH 推导共享密钥 K (session key)
        Note over Client: 用 HP 验证服务器身份<br/>首次连接确认 fingerprint → known_hosts
    end

    rect rgb(220, 252, 231)
        Note over Client,Server: 🟢 阶段三：用户认证 — 试探请求
        Client->>Server: SSH_MSG_USERAUTH_REQUEST<br/>用户名 + 公钥 (无签名)
        Server->>Server: 读取 authorized_keys 逐行比对
        alt 公钥匹配
            Server-->>Client: SSH_MSG_USERAUTH_PK_OK ✅
        else 公钥不匹配
            Server-->>Client: SSH_MSG_USERAUTH_FAILURE ❌ → 回退密码认证
        end
    end

    rect rgb(252, 231, 243)
        Note over Client,Server: 🔴 阶段四：用户认证 — 签名验证
        Client->>Server: SSH_MSG_USERAUTH_REQUEST<br/>用户名 + 公钥 + 签名
        Note over Client: signature = Sign(sk, H(session_id ‖ 请求数据))<br/>session_id 绑定当前会话，防重放攻击
        Server->>Server: Verify(pk, 数据, 签名)
        Server-->>Client: SSH_MSG_USERAUTH_SUCCESS ✅
    end

    Note over Client,Server: 🎉 认证通过，打开 pty 会话
```

### 1.3 多用户场景

多用户只是 `authorized_keys` 文件的排列组合，核心认证逻辑完全一致。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '12px'}}}%%
flowchart LR
    subgraph Server["🖧 服务器文件系统"]
        direction TB
        A["📁 /home/alice/.ssh/authorized_keys<br/>├─ pk_dave.pub ✅<br/>└─ pk_admin.pub ✅"]
        B["📁 /home/bob/.ssh/authorized_keys<br/>└─ pk_bob.pub ✅"]
        C["📁 /home/carol/.ssh/authorized_keys<br/>├─ pk_alice.pub ✅<br/>└─ pk_carol.pub ✅"]
    end

    subgraph Clients["👤 客户端"]
        direction TB
        D["Dave (sk_dave)<br/>→ alice@server ✅<br/>→ bob@server ❌"]
        E["Alice (sk_alice)<br/>→ alice@server ❌<br/>→ carol@server ✅"]
        F["Bob (sk_bob)<br/>→ bob@server ✅<br/>→ carol@server ❌"]
    end

    D -.->|匹配| A
    E -.->|匹配| C
    F -.->|匹配| B

    classDef ok fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef denied fill:#fef2f2,stroke:#dc2626,color:#7f1d1d
```

**关键规则**：
- 同一公钥可授权多个用户（一把私钥登多个账户）
- 一个用户可被多个公钥授权（多设备登录同一账户）
- 认证结果取决于：客户端用的**私钥** 是否在目标用户 `authorized_keys` 中有对应的**公钥**

---

## 二、密钥管理方案对比

### 2.1 方案 A：用户持长期私钥（端到端认证）

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
flowchart LR
    Client["🖥️ 客户端<br/>━━━━━━━━━<br/>🔒 sk_user (本地)<br/>永不离开客户端"] 
    ===|"① SSH 连接<br/>用户私钥签名"| 
    Gateway["🌐 Gateway<br/>━━━━━━━━━<br/>SSH 透传代理<br/>不持有 sk_user<br/>不解密会话内容"]
    ===|"② SSH 转发<br/>用户私钥签名"|
    Container["📦 容器实例<br/>━━━━━━━━━<br/>sshd<br/>authorized_keys<br/>📥 pk_user"]

    classDef clientNode fill:#dbeafe,stroke:#2563eb,color:#1e40af
    classDef gwNode fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef containerNode fill:#dcfce7,stroke:#16a34a,color:#14532d
    
    class Client clientNode
    class Gateway gwNode
    class Container containerNode
```

| 动作 | 内容 |
|------|------|
| **密钥生成** | 用户本地 `ssh-keygen`，私钥永不离身 |
| **公钥部署** | 用户通过 WS/API 将公钥上传 Gateway → Gateway 拉起实例时写入 `authorized_keys` |
| **认证方式** | SSH 透传（TCP 转发或 ProxyJump），Gateway 只转发加密流量不解密 |
| **私钥传输** | **无**，私钥始终在用户本地 |
| **安全等级** | 🔒 **高** — Gateway 无法看到会话内容，即使被攻破也无法伪装用户 |

### 2.2 方案 B：Gateway 动态生成临时密钥

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
sequenceDiagram
    participant Client as 🖥️ 客户端<br/>(无持久密钥)
    participant Gateway as 🌐 Gateway<br/>(密钥生成器)
    participant Instance as 📦 容器实例
    
    rect rgb(219, 234, 254)
        Note over Client,Instance: ① 实例拉起请求
        Client->>Gateway: WS: launch_instance
        Gateway->>Gateway: 生成临时密钥对<br/>(sk_tmp, pk_tmp)
    end
    
    rect rgb(255, 237, 213)
        Note over Gateway,Instance: ② 公钥注入 + SSH 连接
        Gateway->>Instance: 写入 pk_tmp → authorized_keys
        Gateway->>Instance: SSH 连接 (使用 sk_tmp)
    end
    
    rect rgb(220, 252, 231)
        Note over Client,Gateway: ③ 私钥下发
        Gateway-->>Client: WS: 回传 sk_tmp
    end
    
    rect rgb(252, 231, 243)
        Note over Client,Instance: ④ 后续使用
        Client->>Gateway: SSH 连接 (使用 sk_tmp)
        Gateway->>Instance: SSH 转发
    end
```

| 动作 | 内容 |
|------|------|
| **密钥生成** | Gateway 在拉起实例时动态生成临时密钥对 |
| **公钥部署** | Gateway 直接写入实例 `authorized_keys` |
| **私钥回传** | Gateway 通过已有 WS 安全通道将私钥下发给客户端 |
| **私钥传输** | ⚠️ **有** — 私钥经 WS 通道传输一次 |
| **生命周期** | 实例销毁时密钥对同时废弃 |
| **安全等级** | 🔓 **中高** — Gateway 持有私钥副本，可看到会话内容；但密钥一次性，攻击窗口小 |

### 2.3 方案对比总结

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '12px'}}}%%
quadrantChart
    title 方案综合评估
    x-axis 用户操作复杂度低 → 高
    y-axis 安全等级低 → 高
    quadrant-1 理想区间
    quadrant-2 安全优先
    quadrant-3 需优化
    quadrant-4 体验优先
    "方案 A 长期密钥": [0.65, 0.85]
    "方案 B 临时密钥": [0.25, 0.55]
```

| 维度 | 🔑 方案 A (长期密钥) | 🔄 方案 B (临时密钥) |
|------|:---:|:---:|
| 私钥是否离开客户端 | ❌ 否 | ✅ 是（经 WS 传输一次） |
| Gateway 可读会话 | ❌ 否 | ✅ 是 |
| 用户操作复杂度 | 🔴 高（需生成+上传公钥） | 🟢 **低**（无感知） |
| 客户端 SSH 依赖 | ✅ 需要本地安装 | ❌ 不需要 |
| 密钥泄露波及面 | 🔴 大（长期有效） | 🟢 小（单实例、一次性） |
| 适合 TUI 载体 | 🖥️ **本地终端模拟器** | 🌐 纯浏览器（xterm.js） |

---

## 三、TUI 载体分析

### 3.1 方式一：浏览器终端（xterm.js over WS）

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
flowchart LR
    Browser["🌐 浏览器<br/>xterm.js"] 
    ===|"WS 帧<br/>(pty I/O 封装)"| 
    GW["🌐 Gateway<br/>━━━━━━━━<br/>WS ↔ SSH 协议转换"]
    ===|"SSH 连接<br/>(Gateway 持有密钥)"|
    Instance["📦 容器实例<br/>sshd + Claude Code"]

    classDef browserNode fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef gwNode fill:#dbeafe,stroke:#2563eb,color:#1e40af
    classDef instNode fill:#dcfce7,stroke:#16a34a,color:#14532d
    
    class Browser browserNode
    class GW gwNode
    class Instance instNode
```

- SSH 协议终结在 Gateway，Gateway 将 pty 输入输出通过 WS 帧转发
- Gateway 同时作为 SSH 客户端和 WS 服务端
- 密钥在 Gateway 上管理（方案 B 天然匹配）
- 防火墙友好，仅需 443 端口

### 3.2 方式二：本地终端模拟器（Native SSH）

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
flowchart LR
    Term["🖥️ 本地终端<br/>ssh client + sk_user"] 
    ===|"SSH 加密连接<br/>(用户私钥签名)"|
    GW["🌐 Gateway<br/>━━━━━━━━<br/>TCP 层透传<br/>(不解密 SSH)"]
    ===|"SSH 加密连接<br/>(用户私钥签名)"|
    Instance["📦 容器实例<br/>sshd + Claude Code"]

    classDef termNode fill:#e0e7ff,stroke:#4f46e5,color:#312e81
    classDef gwNode fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef instNode fill:#dcfce7,stroke:#16a34a,color:#14532d
    
    class Term termNode
    class GW gwNode
    class Instance instNode
```

- 客户端运行标准 SSH，Gateway 做 ProxyJump 或 TCP 转发
- SSH 连接端到端加密（Gateway 只透传 TCP/SSH 层）
- 密钥在本地（方案 A 天然匹配）

### 3.3 区别与联系

| | 🌐 xterm.js over WS | 🖥️ 本地 SSH 客户端 |
|---|---|---|
| **网络通道** | 复用已有 WS 连接（单端口 443） | 新 TCP 连接（端口 22 或其他） |
| **SSH 对端** | Gateway（Gateway 做 SSH 客户端连实例） | Gateway 透明转发，实际对端是实例 |
| **密钥位置** | Gateway | 用户本地 |
| **WS/SSH 关系** | WS **承载** SSH 的终端 I/O，SSH 对用户透明 | WS 和 SSH **独立并存**，各管各的通道 |
| **防火墙友好** | ✅ 是（仅需 443） | ⚠️ 可能需要开放额外端口 |
| **客户端依赖** | 仅需浏览器 | 需安装 SSH 客户端 |

---

## 四、选定方案：方案 A — 端到端公钥认证

### 4.1 整体拓扑

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
flowchart TB
    subgraph Client["🖥️ 客户端工作站"]
        direction TB
        Browser["🌐 浏览器<br/>━━━━━━━━<br/>WebSocket 控制面<br/>· 统一认证 token<br/>· 实例管理 API<br/>· 公钥上传"]
        Terminal["💻 本地终端模拟器<br/>━━━━━━━━<br/>SSH 数据面<br/>· sk_alice 本地签名<br/>· cc TUI 交互"]
        KeyStore["🔑 ~/.ssh/<br/>━━━━━━━━<br/>sk_alice<br/>pk_alice"]
    end

    subgraph Gateway["🌐 Gateway"]
        direction TB
        WS["📡 WS Server<br/>━━━━━━━━<br/>· 统一认证<br/>· 实例生命周期<br/>· 状态推送"]
        SSHProxy["🔀 SSH 透传<br/>━━━━━━━━<br/>· TCP 端口转发<br/>· ProxyJump"]
        KeyDB["🗄️ 公钥存储<br/>━━━━━━━━<br/>pk_alice → alice"]
    end

    subgraph Container["📦 容器实例"]
        direction TB
        SSHD["🔧 sshd"]
        AK["📥 authorized_keys<br/>pk_alice"]
        CC["🤖 Claude Code<br/>TUI 交互"]
    end

    Browser -->|"① WS: switch cc<br/>上传公钥 / 拉起实例"| WS
    WS -->|"② 注入 pk_alice"| AK
    WS -->|"③ 返回连接信息<br/>{host, port, user}"| Browser
    Browser -.->|"④ 触发 ssh 命令"| Terminal
    Terminal -->|"⑤ SSH 加密<br/>(sk_alice 签名)"| SSHProxy
    SSHProxy -->|"⑥ TCP/SSH 透传<br/>(Gateway 不解密)"| SSHD
    SSHD --> CC

    classDef clientBlock fill:#eff6ff,stroke:#2563eb,color:#1e40af
    classDef gwBlock fill:#fffbeb,stroke:#d97706,color:#78350f
    classDef containerBlock fill:#f0fdf4,stroke:#16a34a,color:#14532d
    classDef browserNode fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef termNode fill:#c7d2fe,stroke:#6366f1,color:#312e81
    classDef keyNode fill:#fef3c7,stroke:#eab308,color:#713f12
    classDef wsNode fill:#e0f2fe,stroke:#0ea5e9,color:#0c4a6e
    classDef proxyNode fill:#fed7aa,stroke:#f97316,color:#7c2d12
    classDef ccNode fill:#dcfce7,stroke:#22c55e,color:#14532d
    
    class Client clientBlock
    class Gateway gwBlock
    class Container containerBlock
    class Browser browserNode
    class Terminal termNode
    class KeyStore keyNode
    class WS wsNode
    class SSHProxy proxyNode
    class KeyDB keyNode
    class AK keyNode
    class CC ccNode
    class SSHD ccNode
```

### 4.2 双通道架构

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
flowchart TB
    subgraph ControlPlane["🟦 控制面 — WebSocket 通道 (持久)"]
        direction LR
        C1["🔐 统一认证 Token 续期"] --- C2["🚀 实例拉起 / 销毁"]
        C2 --- C3["📡 状态同步 / 心跳"] --- C4["📤 公钥上传管理"]
    end
    
    subgraph DataPlane["🟩 数据面 — SSH 通道 (按需)"]
        direction LR
        D1["⌨️ 键盘输入"] --- D2["🔀 Gateway TCP 透传"]
        D2 --- D3["🖥️ PTY 输出渲染"] --- D4["🤖 cc TUI 交互"]
    end
    
    ControlPlane -.->|"通道独立<br/>职责分离"| DataPlane
    
    classDef control fill:#eff6ff,stroke:#2563eb,color:#1e40af
    classDef data fill:#f0fdf4,stroke:#16a34a,color:#14532d
    
    class ControlPlane,C1,C2,C3,C4 control
    class DataPlane,D1,D2,D3,D4 data
```

- **WS 通道**（控制面）：持久长连接，负责统一认证续期、实例生命周期管理、状态推送
- **SSH 通道**（数据面）：按需建立，承载 cc 实例的终端 pty I/O，端到端加密

### 4.3 密钥生命周期

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px', 'fontFamily': 'Segoe UI'}}}%%
flowchart LR
    subgraph Phase1["① 初始化 (一次性)"]
        P1["ssh-keygen<br/>-t ed25519<br/>→ sk + pk"]
    end
    
    subgraph Phase2["② 上传 (一次性)"]
        P2["WS API<br/>POST /api/keys<br/>pk → Gateway"]
    end
    
    subgraph Phase3["③ 注入 (每次拉起)"]
        P3["Gateway 拉实例<br/>pk → authorized_keys"]
    end
    
    subgraph Phase4["④ 连接 (每次会话)"]
        P4["ssh -J gateway<br/>sk 本地签名 → 实例"]
    end
    
    Phase1 --> Phase2 --> Phase3 --> Phase4

    classDef phase fill:#f0f9ff,stroke:#0284c7,color:#0c4a6e
    class Phase1,Phase2,Phase3,Phase4,P1,P2,P3,P4 phase
```

#### 阶段 1：初始化 — 用户生成密钥对

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_cc
# → sk_alice : ~/.ssh/id_ed25519_cc        (权限 600)
# → pk_alice : ~/.ssh/id_ed25519_cc.pub    (权限 644)
```

#### 阶段 2：公钥上传 — 通过 WS 控制面

```http
POST /api/keys
Authorization: Bearer <token>
Content-Type: application/json

{
    "public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... alice@workstation"
}
```

Gateway 将公钥关联到用户身份，持久化存储。用户可管理多把密钥（laptop 一把、desktop 一把）。

#### 阶段 3：实例拉取 — 公钥注入

Gateway 在拉起容器实例时将用户公钥注入至 `authorized_keys`：

| 方式 | 说明 | 推荐度 |
|------|------|:---:|
| **cloud-init** | 通过容器平台的 user-data 注入 | ⭐⭐⭐ |
| **volume mount** | `-v /keys/alice.pub:/home/cc/.ssh/authorized_keys:ro` | ⭐⭐ |
| **SSH 后门注入** | Gateway 先用临时密钥 SSH 进入，写入后删除 | ⭐ |

#### 阶段 4：SSH 连接

```bash
# 直接使用 ProxyJump
ssh -J alice@gateway:2222 cc@<container-ip>

# 或配置 ~/.ssh/config
Host cc
    HostName <container-ip>
    User cc
    ProxyJump alice@gateway:2222
    IdentityFile ~/.ssh/id_ed25519_cc

ssh cc
```

### 4.4 完整时序图

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '12px'}}}%%
sequenceDiagram
    autonumber
    
    box rgb(219, 234, 254) 客户端
        participant B as 🌐 浏览器
        participant T as 💻 本地终端
    end
    
    participant G as 🌐 Gateway
    participant I as 📦 容器实例

    rect rgb(239, 246, 255)
        Note over B,G: 📤 阶段一：公钥上传（一次性操作）
        B->>G: WS: POST /api/keys {public_key}
        G->>G: 存储 pk_alice → alice
        G-->>B: ✅ OK
    end

    rect rgb(255, 247, 237)
        Note over B,I: 🚀 阶段二：实例拉起
        B->>G: WS: switch cc (launch_instance)
        G->>I: 拉取镜像，启动容器
        G->>I: 注入 pk_alice 到 authorized_keys
        I->>I: sshd 启动就绪
        I-->>G: ✅ sshd ready
        G-->>B: 📋 {host, port, user="cc"}
    end

    rect rgb(240, 253, 244)
        Note over B,T: 🔗 阶段三：触发 SSH 连接
        B-->>T: 提示/自动触发 ssh 命令
    end

    rect rgb(253, 242, 248)
        Note over T,I: 🔐 阶段四：SSH 握手（Gateway 透传）
        T->>G: TCP 连接 Gateway:2222
        G->>I: TCP 转发到实例:22
        
        Note over T,I: ECDH 密钥交换（端到端加密）
        T<<->>I: 协商 session key
        
        T->>I: SSH_MSG_USERAUTH_REQUEST (pk_alice)
        I->>I: 查找 authorized_keys<br/>匹配 pk_alice
        I-->>T: SSH_MSG_USERAUTH_PK_OK ✅
        
        T->>I: 签名认证 Sign(sk_alice, 签名数据)
        Note over T: 签名 = Sign(sk, H(session_id ‖ 请求))<br/>防重放攻击
        I->>I: Verify(pk_alice, 数据, 签名)
        I-->>T: SSH_MSG_USERAUTH_SUCCESS ✅
        
        Note over T,I: 🎉 pty 会话建立，cc TUI 开始渲染
    end

    Note over B,G: 📡 WS 通道保持，持续状态同步
```

### 4.5 SSH 透传实现要点

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
flowchart TB
    subgraph I["方式一：TCP 端口转发"]
        direction LR
        T1["🖥️ 客户端<br/>ssh -p 2222"] -->|"TCP"| G1["🌐 Gateway<br/>:2222 → 172.17.0.3:22"]
        G1 -->|"TCP 层转发"| C1["📦 alice 实例"]
    end
    
    subgraph II["方式二：SSH ProxyJump"]
        direction LR
        T2["🖥️ 客户端<br/>ssh -J alice@gateway"] -->|"SSH"| G2["🌐 Gateway<br/>internal-jump 子系统"]
        G2 -->|"动态解析目标"| C2["📦 alice 实例"]
    end

    classDef mode fill:#f0f9ff,stroke:#0284c7,color:#0c4a6e
    class I,II mode
```

#### 方式 I：TCP 端口转发（更简单）

Gateway 为每个实例分配一个转发端口，随实例生命周期动态分配和回收：

| Gateway 端口 | 转发目标 | 所属用户 |
|:---:|---|---|
| `:2222` | → `172.17.0.3:22` | alice |
| `:2223` | → `172.17.0.4:22` | bob |

Gateway 做纯 TCP 层转发，完全不接触 SSH 协议层。

#### 方式 II：SSH ProxyJump（更标准）

```ssh_config
# Gateway sshd 配置 — 自定义子系统方案
Match User *
    ForceCommand internal-jump
```

客户端使用 `ssh -J alice@gateway cc@<container>` 即可。Gateway 上的跳板服务根据用户身份和请求参数，动态解析目标实例地址后建立转发。

### 4.6 优势与代价

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
flowchart LR
    subgraph Pros["✅ 优势"]
        direction TB
        P1["🔒 Gateway 不可见会话内容<br/>防中间人攻击"]
        P2["🔑 密钥永不离身<br/>符合安全最佳实践"]
        P3["📦 长期密钥管理简单<br/>一次配置多次使用"]
    end
    
    subgraph Cons["⚠️ 代价"]
        direction TB
        C1["🖥️ 用户需本地 SSH 客户端"]
        C2["📤 公钥需要上传 + 注入流程"]
        C3["🧹 实例销毁时需清理<br/>authorized_keys 残留"]
    end

    classDef pros fill:#f0fdf4,stroke:#16a34a,color:#14532d
    classDef cons fill:#fef2f2,stroke:#dc2626,color:#7f1d1d
    
    class Pros,P1,P2,P3 pros
    class Cons,C1,C2,C3 cons
```

---

## 五、总结

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '13px'}}}%%
flowchart LR
    A["🌐 浏览器<br/>WS 控制面"] -->|"实例管理<br/>公钥上传"| B["🌐 Gateway<br/>SSH 透传<br/>(不解密)"]
    B -->|"公钥注入<br/>流量转发"| C["📦 容器实例<br/>sshd + cc"]
    D["🖥️ 本地终端<br/>sk 签名"] -->|"SSH 端到端加密"| B
    B -.->|"透传"| C

    classDef nodeA fill:#dbeafe,stroke:#2563eb,color:#1e40af
    classDef nodeB fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef nodeC fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef nodeD fill:#e0e7ff,stroke:#4f46e5,color:#312e81

    class A nodeA
    class B nodeB
    class C nodeC
    class D nodeD
```

| 设计要点 | 说明 |
|----------|------|
| **认证链路** | 浏览器 (WS) → Gateway (透传) → 容器 (sshd)，SSH 端到端加密 |
| **通道分离** | WS = 控制面（实例管理），SSH = 数据面（cc TUI 交互） |
| **密钥策略** | 用户持长期私钥（方案 A），Gateway 不持有私钥、不可见会话内容 |
| **公钥注入** | Gateway 在拉起实例时自动将用户公钥写入 `authorized_keys` |
| **安全边界** | Gateway 即使被攻破也无法解密 SSH 会话或伪装用户身份 |
