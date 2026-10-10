---
title: "仓库助手工作约定"
type: reference
domain: repository
status: active
---

# 仓库助手工作约定

本文件适用于整个仓库。遵循用户当前任务要求；子目录存在更具体的 `AGENTS.md` 时，按其适用范围执行。

## 仓库定位与内容归属

这是以 Markdown 为主的工程知识库，围绕 Agent 和推理基础设施的业务研究、设计与实践，沉淀可复用知识。仓库内容是事实来源，Obsidian、GitHub、Codex、RAG 和 MCP 都是其使用端。

开始任务时阅读 [README.md](README.md) 和目标目录的 README，先检索已有内容，再决定更新或新建。

| 路径 | 内容 |
|---|---|
| `agent/designs/` | Agent 业务设计与决策 |
| `agent/investigations/` | Agent 调研与问题调查 |
| `agent/reports/` | 实践报告与验证结果 |
| `agent/study-notes/` | 尚在推进的学习与源码走读 |
| `infra/` | 推理基础设施业务研究、平台适配与交付历史 |
| `knowledge/agent/` | 可复用的 Agent 架构、概念、集成和产品知识 |
| `knowledge/infra/` | 可复用的推理基础设施、模型与实现知识 |
| `knowledge/foundations/` | 跨领域的硬件和系统基础知识 |
| `templates/` | 文档模板 |
| `.agents/skills/` | 仓库专用助手工作流 |

保留业务文档的背景、约束、调查过程、决策和验证记录。原子知识回答一个可独立检索的问题，不以篇幅长短判断是否需要拆分。

## 编辑与格式

- 默认用中文写作，保留必要的英文术语、API 名称和代码标识符；编辑已有文档时保持其语言和风格。
- 路径使用英文 kebab-case，每篇文档只有一个清晰的一级标题。
- Markdown 文档使用 YAML frontmatter，默认仅包含 `title`、`type`、`domain`、`status`；确有检索或维护需要时再增加字段。
- `type` 可用值：`moc`、`concept`、`architecture`、`design`、`implementation`、`investigation`、`work`、`reference`。
- `domain` 可用值：`repository`、`agent`、`infra`、`foundations`。
- `status` 可用值：`draft`、`active`、`evergreen`、`archived`。未充分验证的内容用 `draft`，已验证但持续演进的内容用 `active`，稳定且边界明确的知识才用 `evergreen`。
- 核心导航使用相对 Markdown 链接，避免依赖 Obsidian 专有语法或本机绝对路径。新增文档时维护最近的 README/MOC，只添加有实际帮助的相关链接。
- 按内容选用 [业务研究模板](templates/business-study.md)、[原子知识模板](templates/atomic-knowledge.md) 或已有专用模板；不要机械保留空章节和占位符。

详细分类与元数据规则见 [repository-conventions.md](.agents/skills/knowledge-distillation/references/repository-conventions.md)。

## 研究与知识沉淀

- 新增或更新名词解释、API 格式、机制及适用边界时，阅读并执行 [atomic-knowledge 技能](.agents/skills/atomic-knowledge/SKILL.md)。简单术语请求保持简洁，不扩展成无关的研究报告。
- 从已完成的调查、设计、修复、PoC 或源码走读提炼知识时，阅读并执行 [knowledge-distillation 技能](.agents/skills/knowledge-distillation/SKILL.md)。探索阶段先记录问题和证据，不提前制造结论。
- 按文件名、标题、中英文名称和常见别名查找已有笔记，优先更新回答同一问题的文档。
- 优先引用官方文档、标准、源代码和论文。涉及会变化的 API、版本、产品行为或外部链接时核实当前来源，说明适用版本和边界。
- 明确区分已验证事实、推断和待验证假设；不编造来源、实验结果或完成状态。
- 从业务文档提炼知识后，维护业务文档与知识笔记的双向相对链接，并更新知识沉淀检查项。没有来源业务文档时，不创建虚假的应用记录。

## 修改范围与历史

- 先检查 `git status --short`，保留用户已有的修改和未跟踪文件，仅修改任务涉及的内容。
- 对已跟踪文件的结构调整使用 `git mv`，同时修复受影响的链接；移动时避免顺带重写技术内容。
- 保留原始业务记录，不因知识提炼而删除调查历史或用后来的结论覆盖当时的决策。
- Git 子模块中的外部仓库不作为本仓库自有笔记改写。不要为了知识分类移动可执行资源。
- 提交、推送及其他发布操作按用户授权执行。

## 验证与交付

文档修改完成后，从仓库根目录运行：

```bash
git diff --check
python .agents/skills/knowledge-distillation/scripts/audit_knowledge.py
```

使用环境中可用的 Python 3 命令（如 `python` 或 `python3`）。检查新增文档的元数据、相对链接和索引入口。修复本次修改引入的问题；已有审计错误单独报告，不扩大修改范围。知识提炼队列警告用于提示后续复核，不代表必须自动生成内容。

交付时简要说明新增或更新的文件、索引与链接维护、验证结果及未解决的问题，并说明是否已提交。只报告实际执行过的验证。
