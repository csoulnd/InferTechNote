---
title: "AgentBox Manager Development, Build, Deployment, and Log Guide"
type: guide
domain: agent
status: active
---

# AgentBox Manager 开发、构建、部署与日志指南

本文面向 AgentBox Manager 的日常开发、联调、出包、部署和问题定位。命令默认从仓库根目录执行；完整部署与镜像构建应在 Linux 目标机或可运行 Linux 容器的 WSL 环境执行。

> 适用仓库：`AgentBox-Manager`。原 `agent-os` 仓库中的 `control-panel/` 已迁移为本仓库根目录下的 `backend/`、`frontend/`、`image_process/`、`image/`、`deploy/` 和 `build/`。本仓库只负责 Manager，不包含 AgentOS Server 的构建与部署。

## 1. 仓库结构与权威入口

| 内容 | 位置 | 说明 |
|---|---|---|
| 项目概览 | `README.md` | Manager 能力简介 |
| 后端 | `backend/README.md`、`backend/pyproject.toml` | FastAPI、配置和测试 |
| 前端 | `frontend/README.md`、`frontend/package.json` | Vue/Vite、代理和质量检查 |
| 镜像构建服务 | `image_process/README.md` | 三方 Agent OCI 镜像构建 |
| 部署 | `deploy/README.md` | 单机/多机部署、端口、镜像和验证 |
| 出包脚本 | `build/build.sh` | Manager tgz 和可选离线镜像包 |

本地 README、脚本和配置文件是行为依据。在线仓库为 `https://gitcode.com/Ascend/AgentBox-Manager`；查看其他分支时，应将链接切换到实际分支。

## 2. 开发模式安装与启动（推荐）

### 2.1 开发模式选择

| 场景 | 推荐方式 | 代码生效方式 |
|---|---|---|
| 前后端日常开发 | PostgreSQL/LiteLLM 等依赖用 Compose，前后端在宿主机运行 | Vite HMR；Uvicorn reload |
| 只开发前端 | 连接一个已运行的 Manager 后端 | Vite HMR |
| 调试三方 Agent 镜像构建 | `image-process` 使用容器运行 | 重建并重启该服务 |
| 验证生产行为 | 构建三个镜像后运行完整 Compose | 重建受影响镜像 |

生产镜像会把后端源码和前端 `dist` 复制到镜像中，因此 `docker compose restart` 不会载入宿主机代码。源码变化应使用热更新开发模式，或重建对应镜像。

### 2.2 前置条件

- Python 3.11+、[`uv`](https://docs.astral.sh/uv/)；
- Node.js 24（与 `image/Dockerfile` 构建阶段一致）和 npm；
- Docker Engine 与 Docker Compose v2；
- Linux/WSL。完整部署脚本还需要 systemd、sudo 和发行版对应的 exporter 制品，详见 `deploy/README.md`。

首次准备：

```bash
git clone https://gitcode.com/Ascend/AgentBox-Manager.git
cd AgentBox-Manager

cd backend && uv sync --extra dev && cd ..
cd frontend && npm ci && cd ..
```

### 2.3 启动开发依赖

复制部署配置并至少填写 PostgreSQL、JWT、管理员和 LiteLLM 密钥：

```bash
cd deploy
cp .env.example .env
openssl rand -hex 32   # 可用于 AGENTOS_JWT_SECRET_KEY
openssl rand -hex 32   # 可用于 LITELLM_KEY_ENCRYPTION_KEY
```

`.env` 的最小必填项：

```dotenv
POSTGRES_USER=agentos
POSTGRES_PASSWORD=<仅字母和数字的密码>
AGENTOS_JWT_SECRET_KEY=<64位十六进制随机值>
AGENTOS_ADMIN_USERNAME=admin
AGENTOS_ADMIN_PASSWORD=<管理员密码>
LITELLM_MASTER_KEY=sk-<自定义密钥>
LITELLM_KEY_ENCRYPTION_KEY=<64位十六进制随机值>
```

从 `deploy/` 启动本地联调所需依赖：

```bash
docker compose up -d postgres litellm victoriametrics loki grafana
docker compose ps
```

`alloy` 只在调试日志采集链路时启动；它依赖 Linux 宿主机日志目录。`image-process` 只在调试三方 Agent 上传/构建时启动，见 2.6。

### 2.4 启动后端（自动 reload）

后端从当前工作目录的 `.env` 读取配置。创建 `backend/.env`，不要直接复用 Compose 里的容器主机名：

```bash
cd backend
cp .env.example .env
```

开发环境至少配置：

```dotenv
AGENTOS_DATABASE_URL=postgresql+asyncpg://agentos:<密码>@127.0.0.1:5432/agentos
AGENTOS_JWT_SECRET_KEY=<与 deploy/.env 一致>
AGENTOS_ADMIN_USERNAME=admin
AGENTOS_ADMIN_PASSWORD=<与 deploy/.env 一致>
LITELLM_ADMIN_URL=http://127.0.0.1:8100
LITELLM_MASTER_KEY=<与 deploy/.env 一致>
LITELLM_KEY_ENCRYPTION_KEY=<与 deploy/.env 一致>
LITELLM_DATABASE_URL=postgresql+asyncpg://agentos:<密码>@127.0.0.1:5432/litellm
AGENTOS_HOME_BASE=/tmp/agentbox-manager/users
LOG_DIR=/tmp/agentbox-manager/logs
```

启动：

```bash
uv sync --extra dev
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

验证：

```bash
curl http://127.0.0.1:8000/health
```

保存 Python 文件后 Uvicorn 会自动重载。若修改依赖声明，需重新执行 `uv sync --extra dev`。

### 2.5 启动前端（HMR）

另开终端：

```bash
cd frontend
VITE_PROXY_TARGET=http://127.0.0.1:8000 npm run dev -- --host 0.0.0.0
```

Windows PowerShell 对应命令：

```powershell
cd frontend
$env:VITE_PROXY_TARGET = "http://127.0.0.1:8000"
npm run dev -- --host 0.0.0.0
```

浏览器使用 Vite 输出的地址（通常为 `http://127.0.0.1:5173`）。`VITE_PROXY_TARGET` 未设置时不会启用 `/api` 代理。保存 Vue/TypeScript/CSS 文件后由 Vite HMR 生效；依赖变化需重新执行 `npm ci`。

### 2.6 调试 image-process

该服务必须访问宿主机 Docker socket，并要求 Docker daemon 中已有 `agent-base` 镜像。推荐仍以容器运行：

```bash
# 仓库根目录
docker build --no-cache -f image_process/base.Dockerfile -t agent-base:1.0 image_process
docker build -f image_process/Dockerfile -t agentos-image-process:latest image_process

cd deploy
docker compose up -d image-process
docker compose exec -T image-process \
  .venv/bin/python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8091/health').read().decode())"
```

若宿主机后端也要调用该容器，注意 `image-process` 只连接 internal 网络且不发布宿主端口。安全的常规联调方式是让后端也运行在完整 Compose 中；若临时发布 8091 端口，应使用仅用于开发、且不提交生产部署包的 Compose override，并把 `backend/.env` 的 `IMAGE_PROCESS_URL` 指向该地址。

`image_process/app` 修改后需重新构建服务镜像：

```bash
docker build -f image_process/Dockerfile -t agentos-image-process:latest image_process
cd deploy && docker compose up -d --force-recreate image-process
```

## 3. 生产镜像联调

从仓库根目录构建：

```bash
docker build -f image/Dockerfile -t agentos-control-panel:latest .
docker build -f image_process/Dockerfile -t agentos-image-process:latest image_process
docker build --no-cache -f image_process/base.Dockerfile -t agent-base:1.0 image_process
```

然后配置并启动完整栈：

```bash
cd deploy
cp .env.example .env   # 已存在时不要覆盖
# 编辑 .env
docker compose up -d
docker compose ps
```

访问 `http://<部署机地址>:8090`。主镜像内 Nginx 提供前端并反向代理端口 8000 的 FastAPI；宿主机只发布 Nginx 的 `FRONTEND_PORT`。

变更后的最小重建范围：

| 变更 | 操作 |
|---|---|
| `frontend/`、`backend/`、`image/` | 重建 `agentos-control-panel`，再 `docker compose up -d --force-recreate agentos` |
| `image_process/app` 或其 Dockerfile | 重建 `agentos-image-process`，再重建 `image-process` 服务 |
| `image_process/base.Dockerfile` | 重建 `agent-base` |
| `deploy/docker-compose.yml`、Nginx/监控配置 | `docker compose up -d`；必要时重建相关服务 |
| `.env` | `docker compose up -d --force-recreate <受影响服务>` |

## 4. 构建发布包

`build/build.sh` 生成 `build/dist/AgentOS-Manager-<arch>.tgz`，构建会重建其 staging/output，勿在 `build/dist/` 放置手工文件。

```bash
# 只打包 deploy/，daily 版本
./build/build.sh

# release 版本（版本号取脚本内 RELEASE_VERSION）
./build/build.sh --mode release

# 构建三个镜像后打包
./build/build.sh --build-images

# 构建镜像并将 docker save 结果纳入离线包
./build/build.sh --build-images --include-images

# 显式统一镜像 tag
./build/build.sh --build-images --image-tag latest
```

包内 `deploy/VERSIONS` 记录 Manager 与三个镜像的版本；部署脚本会把镜像 tag 同步到 `.env`。发布前检查：

```bash
tar tzf build/dist/AgentOS-Manager-$(uname -m).tgz | head
```

并保存构建模式、参数、架构和 Git commit。

## 5. 安装部署

### 5.1 单机 master

目标机解压发布包后：

```bash
tar xzf AgentOS-Manager-$(uname -m).tgz
cd AgentOS-Manager/deploy

sudo bash deploy.sh install
cd ~/.agentos/.agent-manager
sudo bash deploy.sh up
sudo bash deploy.sh status
```

`install` 会把部署目录复制到 `~/.agentos/.agent-manager`。此后的 `up/down/restart/status/uninstall` 必须从安装目录执行。

### 5.2 多机

master：

```bash
sudo bash deploy.sh install --mode multi --workers 192.168.1.11,192.168.1.12
```

每台 worker：

```bash
sudo bash deploy.sh install --role worker --master-ip 192.168.1.10
```

worker 不运行 Manager Compose，只安装 exporter，并将 Alloy 日志发送到 master。详细的镜像、exporter、NPU 和 SkillHub 前置条件以 `deploy/README.md` 为准。

### 5.3 生命周期与清理

```bash
cd ~/.agentos/.agent-manager
sudo bash deploy.sh restart
sudo bash deploy.sh down
sudo bash deploy.sh status
sudo bash deploy.sh uninstall
```

普通 `uninstall` 默认保留数据和 `.env`。`uninstall --clean` 会删除 Compose 数据卷、`.env` 和安装目录，仅在明确要清空环境时使用。

## 6. 默认端口

端口可由 `deploy/.env` 覆盖；以实际 `.env` 和 `docker compose config` 为准。

| 服务 | 默认宿主端口 | 说明 |
|---|---:|---|
| Manager Web/Nginx | 8090 | 对外入口，容器内 80 |
| Vite 开发服务器 | 5173 | 仅开发模式 |
| FastAPI 开发服务器 | 8000 | 仅宿主机开发模式；生产不直接发布 |
| PostgreSQL | 5432 | 仅绑定 `127.0.0.1` |
| LiteLLM | 8100 | 容器内 4000 |
| VictoriaMetrics | 8428 | 仅绑定 `127.0.0.1` |
| Grafana | 8093 | 调试直出；正常经 `/grafana/` 访问 |
| Loki | 8096 | 日志 API |
| Alloy HTTP | 12345 | 仅绑定 `127.0.0.1` |
| node_exporter | 8091 | 宿主机 systemd 服务 |
| npu-exporter | 8092 | 宿主机 systemd 服务 |
| node-service | 8101 | 可选宿主机 systemd 服务 |
| image-process | 无 | 容器内 8091，仅 `image-build` internal 网络 |
| SkillHub | 8098 | 可选组件 |

注意：node_exporter 默认宿主端口和 image-process 容器端口均为 8091，但后者没有发布到宿主机，不冲突。

## 7. 日志与故障定位

### 7.1 开发模式

- 后端异常：查看运行 Uvicorn 的终端；
- 前端构建/运行异常：查看 Vite 终端和浏览器 Console/Network；
- API 未到达后端：先确认 `VITE_PROXY_TARGET`，再查浏览器请求地址；
- 数据库/LiteLLM 异常：从 `deploy/` 查看对应 Compose 日志。

```bash
cd deploy
docker compose logs -f --tail 200 postgres litellm
docker compose logs -f --tail 200 image-process
```

### 7.2 完整部署

```bash
cd ~/.agentos/.agent-manager
sudo bash deploy.sh status
docker compose ps
docker compose logs --since 15m --timestamps
docker compose logs -f --tail 200 agentos image-process
docker compose logs -f --tail 200 postgres litellm
docker compose logs -f --tail 200 victoriametrics loki alloy grafana
```

宿主机服务：

```bash
sudo journalctl -u node_exporter -n 200 --no-pager
sudo journalctl -u npu-exporter -n 200 --no-pager
sudo journalctl -u agentos-node-service -n 200 --no-pager
```

常用文件目录：

| 内容 | 默认位置 |
|---|---|
| Manager 用户/业务日志 | `${AGENTOS_BASE:-/home/agentos}/logs` |
| 聚合组件日志 | `/var/log/agentos` |
| JiuwenBox 日志 | `${JIUWENBOX_LOG_DIR:-/tmp/jiuwenbox}` |
| 日志导出 | 安装目录下 `exports/` 挂载到容器 `/var/log/agentos_exports` |

优先按故障时间窗查看 100～200 行日志，并用 request/trace/task ID 关联。分享日志前必须脱敏 `.env`、API key、JWT/OAuth secret、密码、请求正文和用户数据。

## 8. 提交前最小检查

```bash
# 后端
cd backend
uv sync --extra dev
uv run pytest

# image-process
cd ../image_process
uv sync --extra dev
uv run pytest

# 前端
cd ../frontend
npm ci
npm run check
npm run build

# 脚本与 Compose（Linux/WSL）
cd ..
bash -n build/build.sh
bash -n deploy/deploy.sh
docker compose -f deploy/docker-compose.yml --env-file deploy/.env config >/dev/null
```

涉及部署行为时，还应在干净测试机验证 `install → up → status → restart → down → uninstall`，并确认普通卸载与 `--clean` 的数据保留/删除行为符合预期。
