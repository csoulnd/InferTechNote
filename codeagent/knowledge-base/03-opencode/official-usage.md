# OpenCode 官方用法

## P0 必读

### [OpenCode Server](https://opencode.ai/docs/server/)
- **type**: standard | **priority**: P0 | **status**: todo
- **notes**: `opencode serve --port 4096` 独立部署；Agent OS 容器首选

### [OpenCode CLI](https://opencode.ai/docs/cli/)
- **type**: standard | **priority**: P0 | **status**: todo
- **notes**: `run` / `attach <url>` / `run --attach` / `acp`

### [Agents (Build / Plan)](https://opencode.ai/docs/agents/)
- **type**: standard | **priority**: P0 | **status**: todo
- **notes**: Build 全权限 vs Plan 只读；子 Agent 配置

### [Permissions](https://opencode.ai/docs/permissions/)
- **type**: standard | **priority**: P1 | **status**: todo
- **notes**: `permission.edit`、`permission.bash` 等

### [OpenCode Web](https://opencode.ai/docs/web/)
- **type**: standard | **priority**: P2 | **status**: todo

## 本地验收

- [ ] `opencode serve` + 另一终端 `opencode attach http://localhost:4096`
- [ ] `opencode run --no-tui "hello"`
- [ ] `opencode acp`（stdio JSON-RPC）

## 远程 Server 示例

```bash
OPENCODE_SERVER_PASSWORD=secret opencode serve --port 4096 --hostname 0.0.0.0
opencode attach http://<remote-ip>:4096
```
