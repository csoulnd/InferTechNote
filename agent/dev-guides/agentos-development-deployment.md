---
title: "AgentOS Development, Build, Deployment, and Log Guide"
type: guide
domain: agent
status: active
---

# AgentOS 开发、构建、部署与日志指南

本文面向 AgentOS 日常开发和问题定位，覆盖源码快速迭代、发布包构建、Manager/Server 部署、常用端口和日志排查。

> 适用仓库：`agent-os`。命令默认从仓库根目录执行；部署命令默认在 Linux 目标机执行。

## 1. 权威说明入口

| 内容 | AgentOS 仓库内文档 | 在线链接 |
|---|---|---|
| 仓库与 submodule | `README.md` | [README](https://gitcode.com/openJiuwen/agent-os/blob/master/README.md) |
| 发布包构建 | `build/README.md` | [build/README](https://gitcode.com/openJiuwen/agent-os/blob/master/build/README.md) |
| Server 部署 | `deploy/README.md` | [deploy/README](https://gitcode.com/openJiuwen/agent-os/blob/master/deploy/README.md) |
| Manager 部署 | `control-panel/deploy/README.md` | [control-panel/deploy/README](https://gitcode.com/openJiuwen/agent-os/blob/master/control-panel/deploy/README.md) |
| Manager 后端 | `control-panel/backend/README.md` | [backend/README](https://gitcode.com/openJiuwen/agent-os/blob/master/control-panel/backend/README.md) |
| Manager 前端 | `control-panel/frontend/README.md` | [frontend/README](https://gitcode.com/openJiuwen/agent-os/blob/master/control-panel/frontend/README.md) |

若分支不是 `master`，应切换在线链接到实际分支；本地仓库中的 README 和脚本始终优先。

## 2. 修改代码后如何快速生效

### 2.1 为什么目前重启容器不生效

生产镜像通过 `control-panel/image/Dockerfile` 将前端构建结果和后端源码 `COPY` 到镜像。`docker compose restart` 只重启旧容器，既不会重新读取宿主机源码，也不会重新构建前端静态文件。因此：

- 代码只存在宿主机、未挂载进容器时，重启无效；
- bind mount 挂载源码后，Python 后端可通过 Uvicorn reload 自动生效；
- 前端应运行 Vite dev server 获取 HMR，不能依赖生产 Nginx 中的旧 `dist`；
- 依赖、Dockerfile、系统软件或前端生产包变化仍需重新安装或重建镜像。

### 2.2 推荐：依赖用容器，前后端在宿主机开发

这是改动最少、调试体验最好的方式。先只启动数据库、LiteLLM、VictoriaMetrics 等依赖：

```bash
cd control-panel/deploy
cp .env.example .env
# 按 README 填写必需变量
docker compose up -d postgres litellm victoriametrics loki grafana
```

启动后端：

```bash
cd control-panel/backend
uv sync --extra dev

export AGENTOS_DATABASE_URL='postgresql+asyncpg://<user>:<password>@127.0.0.1:5432/agentos'
export LITELLM_ADMIN_URL='http://127.0.0.1:8100'
export LITELLM_MASTER_KEY='<key>'
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动前端：

```bash
cd control-panel/frontend
npm ci
npm run dev -- --host 0.0.0.0
```

前端默认由 Vite 提供热更新。若 API 地址不是前端当前配置的默认值，按 `vite.config.*` 或项目环境变量调整代理目标到 `http://127.0.0.1:8000`。

这种方式下，前端保存即生效，后端 `.py` 文件保存后 Uvicorn 自动 reload，无需重建镜像或手工重启。

### 2.3 备选：开发专用 Compose override

若开发必须全部运行在容器内，新增一个不进入生产部署包的 `docker-compose.dev.yml`，核心原则是：

```yaml
services:
  agentos-backend-dev:
    volumes:
      - ../backend:/workspace/backend
    working_dir: /workspace/backend
    command: uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  agentos-frontend-dev:
    volumes:
      - ../frontend:/workspace/frontend
      - frontend-node-modules:/workspace/frontend/node_modules
    working_dir: /workspace/frontend
    command: npm run dev -- --host 0.0.0.0

volumes:
  frontend-node-modules:
```

实际使用前还要补齐开发镜像、端口、环境变量、网络和前端 API 代理。不要直接覆盖生产 `agentos` 服务的 `/opt/agentos/backend`，因为该容器同时管理 Nginx 和 Uvicorn，进程生命周期及依赖目录容易互相干扰。

### 2.4 Server 组件快速迭代

Server 发布包安装的是 wheel。修改 `jiuwenswarm/`、`yuanrong/` 或 `agent-protocol/` 源码后，只执行 `./deploy/agentos.sh restart` 不会更新已安装包。开发机应对正在修改的 Python 项目使用 editable install：

```bash
python3.11 -m pip install -e ./jiuwenswarm
# 其他项目按其 pyproject.toml/setup.py 所在目录执行相同操作
```

随后按变更范围重启：

```bash
cd deploy
sudo ./agentos.sh restart
```

若只改一个 systemd 托管组件，可缩短反馈周期：

```bash
sudo systemctl restart agent-registry
sudo systemctl restart agentos-executor
sudo systemctl restart jiuwenbox
sudo systemctl restart 'jiuwenswarm-gateway*' 'jiuwenswarm-web*'
```

先用 `systemctl list-units 'jiuwenswarm-*'` 确认动态生成的实际 unit 名。无 systemd 环境应使用对应模块的 `down/up`，不要仅 `kill` 后假定进程会自动拉起。

必须重新构建或重新安装的情况：依赖声明变化、C/C++/Rust 扩展、RPM、镜像基础层、前端生产静态文件、wheel 打包内容、部署脚本安装阶段生成的 unit/config。

## 3. 编译与出包

### 3.1 初始化源码

```bash
git clone --recurse-submodules https://gitcode.com/openJiuwen/agent-os.git
cd agent-os
git submodule update --init --recursive
```

### 3.2 构建发布包

前置条件为 Bash 4.3+、`curl` 或 `wget`、可访问 OBS/GitCode。建议在 Linux 或 WSL 中执行：

```bash
# daily，默认模式
./build/build.sh daily

# release 示例
./build/build.sh release \
  --cp-tag cp311 \
  --yuanrong-release-version 0.9.0 \
  --jiuwenswarm-release-version 0.2.2 \
  --jiuwenswarm-release-git-tag JiuwenSwarm0.2.2
```

网络不稳定时：

```bash
./build/build.sh daily --download-jobs 1
```

产物位于 `build/dist/`：

| 产物 | 用途 |
|---|---|
| `AgentOS-Manager.tgz` | Manager Compose、配置和部署脚本 |
| `AgentOS-Server-<arch>.tgz` | Server wheels、RPM 依赖和统一部署目录 |
| `AgentOS-Client.tgz` | 客户端 TUI wheels |

构建会清理 `build/dist/`。发布前至少检查文件存在、目标架构正确、压缩包可解压，并保存构建模式、参数、主仓 commit 和 submodule commit。

## 4. 部署

### 4.1 Manager

目标机准备 `AgentOS-Manager.tgz` 以及 README 列出的业务镜像和公共镜像，然后：

```bash
tar xzf AgentOS-Manager.tgz
cd AgentOS-Manager

sudo bash deploy.sh install
cd ~/.agentos/.agent-manager
sudo bash deploy.sh up
sudo bash deploy.sh status
```

日常操作必须从安装目录执行：

```bash
cd ~/.agentos/.agent-manager
sudo bash deploy.sh restart
sudo bash deploy.sh status
sudo bash deploy.sh down
```

多机时，master 示例：

```bash
sudo bash deploy.sh install --mode multi --workers 192.168.1.11,192.168.1.12
```

worker 示例：

```bash
sudo bash deploy.sh install --role worker --master-ip 192.168.1.10
```

`uninstall --clean` 会删除数据卷、`.env` 和安装目录，不应作为普通重部署命令。

### 4.2 Server

```bash
tar xzf AgentOS-Server-$(uname -m).tgz
cd AgentOS-Server/deploy

# 全节点安装 wheel/RPM
sudo ./agentos.sh install

# etcd 节点初始化；多机应先确保全部 etcd 节点达到 quorum
sudo ./agentos.sh init

# 启动应用组件
sudo ./agentos.sh up
sudo ./agentos.sh status
```

生命周期为：

```text
install → init → up ⇄ down → deinit → uninstall
```

`restart` 等价于应用层 `down + up`，不会重装 wheel，也不会重启 etcd。`init` 会按脚本语义清理历史 etcd 数据后 bootstrap，不能把它当普通重启使用。

## 5. 常见端口表

端口以当前脚本默认值为准，部署环境的 `.env` 和模块环境变量可覆盖。

### Manager

| 服务 | 默认端口 | 暴露范围/说明 |
|---|---:|---|
| AgentOS Web/Nginx | 8090 | 对外入口，容器内 80 |
| PostgreSQL | 5432 | 默认仅 `127.0.0.1` |
| LiteLLM | 8100 | 容器内 4000 |
| VictoriaMetrics | 8428 | 默认仅 `127.0.0.1` |
| Grafana | 8093 | 调试直出；正常从 `/grafana/` 访问 |
| Loki | 8096 | 日志存储/API |
| Alloy HTTP | 12345 | 默认仅 `127.0.0.1` |
| node_exporter | 8091 | 宿主机 systemd 服务 |
| npu-exporter | 8092 | 宿主机 systemd 服务 |
| node-service | 8101 | 可选宿主机 systemd 服务 |
| SkillHub 前端 | 8098 | 可选 |
| MinIO S3 API | 8099 | SkillHub 可选 |

### Server

| 服务 | 默认端口/地址 | 说明 |
|---|---:|---|
| etcd client | 32379 | `YR_ETCD_CLIENT_PORT` |
| etcd peer | 32380 | 集群 peer |
| A2X registry | 4003 | `A2X_REGISTRY_PORT` |
| JiuwenSwarm web | 19000 | `WEB_PORT` |
| JiuwenSwarm gateway | 19001 | `GATEWAY_PORT` |
| JiuwenSwarm static | 5173 | `WEB_STATIC_PORT` |
| MooseFS metalogger | 9419 | master 配置固定值 |
| MooseFS master | 9420 | `MFS_MASTER_PORT` |
| MooseFS client | 9421 | `MFS_CLIENT_PORT` |
| MooseFS chunkserver | 9422 | `MFS_CHUNK_PORT` |
| JiuwenBox | Unix socket | 默认 `unix:///run/jiuwenbox/jiuwenbox.sock`，非 TCP |
| YuanRong | 动态端口 | 以 `${YR_LOG_DIR_PREFIX}/latest/session.json` 和状态输出为准 |

检查监听：

```bash
sudo ss -lntup
sudo ss -lx | grep jiuwenbox
```

## 6. 开发日志定位指南

### 6.1 先缩小范围

推荐顺序：

1. 用统一状态命令确认是哪一层失败；
2. 看该服务最近 100～200 行日志，不要一开始导出全部日志；
3. 用请求时间、用户、任务 ID、trace/request ID 交叉检索；
4. 确认端口、进程、依赖健康，再确认业务异常；
5. 多机问题同时检查发起节点、目标节点和 ingress/master 节点的时钟与日志。

### 6.2 Manager 日志

```bash
cd ~/.agentos/.agent-manager
sudo bash deploy.sh status
docker compose ps

# 所有容器，最近 15 分钟
docker compose logs --since 15m --timestamps

# 指定组件并持续跟踪
docker compose logs -f --tail 200 agentos
docker compose logs -f --tail 200 image-process
docker compose logs -f --tail 200 postgres litellm
docker compose logs -f --tail 200 victoriametrics loki alloy grafana
```

宿主机文件日志：

| 内容 | 默认位置 |
|---|---|
| AgentOS 用户/业务日志 | `${AGENTOS_BASE:-/home/agentos}/logs` |
| AgentOS 聚合组件日志 | `/var/log/agentos` |
| JiuwenBox 日志 | `${JIUWENBOX_LOG_DIR:-/tmp/jiuwenbox}` |
| node-service | `/var/log/agentos/agentos-node-service` |

Exporter/node-service：

```bash
sudo journalctl -u node_exporter -n 200 --no-pager
sudo journalctl -u npu-exporter -n 200 --no-pager
sudo journalctl -u agentos-node-service -n 200 --no-pager
sudo journalctl -u agentos-node-service -f
```

若 UI 有报错但后端没有请求，先查浏览器 Network/Console 和 Nginx/`agentos` 容器日志；若后端请求存在但模型调用失败，再查 LiteLLM 和推理服务；若日志面板无数据，依次检查 Alloy → Loki → Grafana 数据源。

### 6.3 Server 日志

```bash
cd AgentOS-Server/deploy
sudo ./agentos.sh status

sudo journalctl -u agentos-etcd -n 200 --no-pager
sudo journalctl -u agent-registry -n 200 --no-pager
sudo journalctl -u agentos-executor -n 200 --no-pager
sudo journalctl -u jiuwenbox -n 200 --no-pager
sudo journalctl -u 'jiuwenswarm-gateway*' -n 200 --no-pager
sudo journalctl -u 'jiuwenswarm-web*' -n 200 --no-pager
sudo journalctl -u conchd -n 200 --no-pager
sudo journalctl -u moosefs-master -u moosefs-chunkserver -u moosefs-client -n 200 --no-pager
```

主要文件日志：

| 组件 | 默认位置 |
|---|---|
| YuanRong | `/var/log/agentos/yr_sessions/latest/`；实际根由 `YR_LOG_DIR_PREFIX` 控制 |
| A2X registry（nohup/文件日志） | `/var/log/agentos/a2x-registry.log` |
| Conch（nohup） | `/var/log/conchd.log` |
| JiuwenBox | 运行目录中的 `jiuwenbox.log`，通常同时可用 `journalctl -u jiuwenbox` |

```bash
sudo find /var/log/agentos/yr_sessions/latest -type f -maxdepth 2 -print
sudo tail -F /var/log/agentos/a2x-registry.log
sudo tail -F /var/log/conchd.log
```

### 6.4 常见故障到日志的映射

| 现象 | 优先查看 |
|---|---|
| 页面 502/接口无响应 | `agentos` 容器、后端 Uvicorn、PostgreSQL |
| 模型调用失败 | `agentos` → `litellm` → 实际推理服务 |
| Agent 镜像构建失败 | `image-process`、Docker daemon、`${AGENTOS_BASE}/images` 空间 |
| Grafana 无指标 | exporter → VictoriaMetrics targets → Grafana datasource |
| Grafana 无日志 | 源日志文件 → Alloy → Loki → Grafana datasource |
| Server `up` 提示 etcd 不可达 | `agentos-etcd`、32379 端口、`config.yaml` 节点列表、防火墙 |
| registry 不可用 | ingress VIP 持有节点、`agent-registry`、etcd、4003 端口 |
| JiuwenSwarm gateway/web 不可用 | 对应动态 systemd unit、19001/19000/5173 |
| 沙箱启动失败 | `jiuwenbox`、`conchd`、Unix socket、policy 和权限 |
| YuanRong task/executor 失败 | `agentos-executor` 与 `yr_sessions/latest` 下对应 session 日志 |

### 6.5 采集一个可分享的最小诊断包

分享前必须检查并脱敏 `.env`、API key、JWT secret、密码、OAuth secret、请求正文和用户数据。建议只收集故障时间窗：

```bash
mkdir -p /tmp/agentos-diag
docker compose ps > /tmp/agentos-diag/compose-ps.txt
docker compose logs --since 30m --timestamps > /tmp/agentos-diag/compose.log 2>&1
sudo journalctl --since '-30 min' \
  -u agentos-etcd -u agent-registry -u agentos-executor -u jiuwenbox -u conchd \
  --no-pager > /tmp/agentos-diag/server-journal.log
sudo ss -lntup > /tmp/agentos-diag/listeners.txt
```

不要直接打包 `.env` 或整个用户日志目录。脱敏后再压缩并分享。

## 7. 开发完成前的最小检查

```bash
# 后端
cd control-panel/backend
uv run pytest

# 前端
cd control-panel/frontend
npm run check

# 部署脚本语法（Linux/WSL）
bash -n deploy/agentos.sh
bash -n control-panel/deploy/deploy.sh
```

涉及部署行为时，还应在干净测试机验证 `install → init/up → status → restart → down`，并确认数据保留/清理行为符合预期。
