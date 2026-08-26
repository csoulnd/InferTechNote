---
title: "工程工具基础"
type: moc
domain: foundations
status: active
---

# 工程工具基础

面向日常开发与远程环境操作的基础工具知识。目标不是一次收录所有参数，而是先建立正确心智模型，再随真实使用逐步加入命令、场景和排障记录。

## 当前学习路径

1. [Git 基础与常用命令](git-basics.md)
   - 理解工作区、暂存区、本地仓库与远端仓库。
   - 掌握查看、暂存、提交、分支、同步和安全撤销。
   - 后续按实际使用追加命令，不把文档扩成无场景的命令大全。
2. [SSH 公钥免密登录](ssh-key-auth.md)
   - 理解公钥认证，而不是简单记住复制命令。
   - 完成密钥生成、服务端授权、客户端别名与权限排障。
3. [Docker 基础与常用语法](docker-basics.md)
   - 理解镜像、容器、卷、网络和 Registry。
   - 掌握 `docker run`、Dockerfile 与 Compose 的基本工作流。

## 维护规则

### Git 命令如何逐步添加

每次遇到新命令，先判断它是否满足以下任一条件：

- 在不同项目中重复使用。
- 解决了一个可复现的问题。
- 容易和相邻命令混淆。
- 具有数据丢失或历史改写风险，需要保留边界说明。

新增时不要只写命令名，至少记录：

```markdown
### `git <command>`：一句话用途

- 场景：为什么需要它。
- 命令：最小可运行示例。
- 结果：它修改工作区、暂存区、提交历史还是远端？
- 风险：能否撤销，是否会覆盖内容或改写历史？
- 相关：与哪个相近命令容易混淆？
```

优先把命令加入现有主题；只有当一个主题已经能独立回答问题时，才拆成新文件。例如：

- `merge`、`rebase`、`cherry-pick` 可逐步形成“Git 历史整合”。
- `reset`、`restore`、`revert` 可形成“Git 撤销模型”。
- `worktree`、`submodule` 可各自形成独立知识点。

### 后续候选知识点

按实际需求选择，不提前创建空文件：

- Linux Shell：路径、权限、管道、重定向、进程与环境变量。
- HTTP/HTTPS：请求方法、状态码、Header、TLS 与代理。
- 网络排障：DNS、端口、路由，以及 `curl`、`ss`、`lsof`。
- 文本处理：`rg`、`sed`、`awk`、`jq`。
- 构建与包管理：Make、npm/pnpm、Python venv/pip。
- Kubernetes：在熟悉 Docker 网络、卷与镜像之后再学习。

## 相关知识

- [底层隔离与 OCI/Docker](../../agent/concepts/infrastructure/01-sandbox-oci-docker.md)
- [SSH Channel 接入](../../agent/integration/ssh-channel.md)

