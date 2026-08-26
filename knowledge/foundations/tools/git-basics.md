---
title: "Git 基础与常用命令"
type: concept
domain: foundations
status: draft
---

# Git 基础与常用命令

## 核心问题

如何理解 Git 的基本状态模型，并安全完成日常查看、提交、分支切换和远端同步？

## 简要结论

Git 的核心不是背命令，而是理解内容在四个位置之间流动：工作区、暂存区、本地仓库和远端仓库。执行命令前先判断“我要读取或修改哪一层”，就能减少误操作。

```mermaid
flowchart LR
    WT[工作区<br/>正在编辑的文件] -->|git add| IDX[暂存区 / Index<br/>下次提交快照]
    IDX -->|git commit| LOCAL[本地仓库<br/>Commit 历史]
    LOCAL -->|git push| REMOTE[远端仓库]
    REMOTE -->|git fetch| LOCAL
    LOCAL -->|git restore --staged| IDX
    IDX -->|git restore| WT
```

## 1. 首次配置与创建仓库

```bash
# 查看版本
git --version

# 配置提交身份；--global 表示当前用户默认值
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# 查看配置及来源
git config --list --show-origin

# 在当前目录创建仓库
git init

# 克隆已有仓库
git clone <repository-url>
```

`user.name` 和 `user.email` 写入提交作者信息，不是登录远端平台的账号密码。远端认证通常使用 SSH key、HTTPS token 或凭据管理器。

## 2. 每次操作前先观察

```bash
# 查看工作区与暂存区状态
git status
git status --short

# 查看尚未暂存的变化：工作区 vs 暂存区
git diff

# 查看已经暂存的变化：暂存区 vs HEAD
git diff --staged

# 查看简洁提交历史
git log --oneline --graph --decorate --all

# 查看某次提交
git show <commit>
```

推荐养成固定顺序：`status → diff → add → diff --staged → commit`。它比直接 `git add . && git commit` 更容易发现密钥、临时文件或无关改动。

## 3. 暂存与提交

```bash
# 暂存指定文件当前内容
git add path/to/file

# 交互式选择修改块，适合拆分提交
git add -p

# 暂存当前目录下的新增、修改和删除
git add -A

# 创建提交
git commit -m "docs: add Git basics"

# 修改最近一次提交信息，或补入刚忘记暂存的内容
git commit --amend
```

关键点：`git add` 保存的是执行当时的文件内容。之后继续修改同一文件，新变化不会自动进入暂存区，需要再次 `git add`。

一个好的提交应只表达一个完整意图。提交信息建议使用祈使或结果描述，例如 `fix: handle empty config`，不要只写 `update`。

## 4. 分支操作

```bash
# 查看本地分支；-a 同时显示远端跟踪分支
git branch
git branch -a

# 创建并切换到新分支
git switch -c feature/my-change

# 切换已有分支
git switch main

# 合并指定分支到当前分支
git merge feature/my-change

# 删除已合并的本地分支
git branch -d feature/my-change
```

`git switch` 专门用于切换分支，语义比同时承担多种职责的 `git checkout` 更清晰。合并前先确认当前分支，因为 `git merge X` 的含义是“把 X 合并进当前分支”。

## 5. 远端同步

```bash
# 查看远端名称和 URL
git remote -v

# 添加远端
git remote add origin <repository-url>

# 只下载远端引用和对象，不自动修改当前分支
git fetch origin

# 查看本地与远端差异
git log --oneline HEAD..origin/main
git log --oneline origin/main..HEAD

# 拉取并整合当前分支的上游变化
git pull

# 首次推送并设置上游分支
git push -u origin feature/my-change

# 后续推送
git push
```

`git pull` 大致等于 `git fetch` 加一次整合操作（merge 或 rebase，取决于配置）。学习阶段建议先 `fetch`、检查差异，再明确选择如何整合。

## 6. 临时保存

```bash
# 临时保存已跟踪文件的未提交修改
git stash push -m "wip: interrupted work"

# 连同未跟踪文件一起保存
git stash push -u -m "wip: include new files"

# 查看列表
git stash list

# 应用最近一项但保留 stash 记录
git stash apply

# 应用最近一项并在成功后移除记录
git stash pop
```

Stash 适合短期切换上下文，不应替代有意义的小提交。应用前先确认所在分支，冲突仍需人工解决。

## 7. 基础撤销

```bash
# 丢弃某文件尚未暂存的修改
git restore path/to/file

# 将文件移出暂存区，但保留工作区修改
git restore --staged path/to/file

# 创建一个新提交，反向抵消已有提交
git revert <commit>
```

三组相近命令的边界：

| 命令 | 主要修改对象 | 基础阶段的使用建议 |
|---|---|---|
| `restore` | 工作区或暂存区文件 | 用于文件级恢复；丢弃工作区修改前必须看 diff |
| `revert` | 新增反向提交 | 已推送历史优先使用，保留可审计历史 |
| `reset` | 暂存区、分支指针，某些模式还会改工作区 | 暂不作为基础撤销手段；理解三棵树后再使用 |

不要在不了解影响时执行 `git reset --hard`、`git clean -fd` 或强制推送；它们可能让未保存内容难以恢复。

## 8. `.gitignore`

`.gitignore` 用于避免未跟踪文件被加入版本库，例如构建产物、缓存和本地环境文件：

```gitignore
.env
node_modules/
dist/
*.log
```

它不会自动停止跟踪已经提交的文件。密钥一旦提交，即使后来删除或加入 `.gitignore`，仍可能存在于历史中，应立即轮换密钥并按团队流程清理历史。

## 命令逐步追加区

后续遇到真实场景时，优先按以下主题追加：

| 主题 | 候选命令 | 何时添加 |
|---|---|---|
| 精准提交 | `add -p`、`commit --fixup` | 需要拆分混合修改时 |
| 历史整合 | `rebase`、`cherry-pick` | 需要整理或移植提交时 |
| 排查问题 | `bisect`、`blame`、`log -S` | 定位引入问题的提交时 |
| 并行工作 | `worktree` | 同时维护多个分支时 |
| 外部仓库 | `submodule` | 项目真实采用子模块时 |

每次新增遵循 [工程工具基础的维护模板](README.md#git-命令如何逐步添加)。

## 适用边界

- 本文只覆盖单仓库的日常基础，不展开 rebase、子模块、Git LFS 与服务端管理。
- 团队可能约定 merge、rebase、签名提交或保护分支策略，应以项目规范为准。
- 任何会丢弃工作区内容、删除未跟踪文件、改写已发布历史的命令，都应先备份并确认精确目标。

## 相关知识

- [SSH 公钥免密登录](ssh-key-auth.md)

## 参考资料

- [Git Reference](https://git-scm.com/docs)
- [Git User Manual](https://git-scm.com/docs/user-manual)
- [Git Cheat Sheet](https://git-scm.com/cheat-sheet.pdf)

