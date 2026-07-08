# Claude Code 泄露源码分析索引

> 2026 年 3 月 `@anthropic-ai/claude-code@2.1.88` npm 包误带 source map，社区产出结构化分析。**不镜像原始源码**，优先读下列分析文档。

## 首选：逐章架构分析

### [0xE1337/decode-claude-code-analysis](https://github.com/0xE1337/decode-claude-code-analysis)
- **type**: analysis | **priority**: P0 | **status**: todo

| 章节 | 路径 | 重点 | 对应 notes |
|------|------|------|------------|
| 00 | Entry & Startup | 4 阶段启动、import 并行 | — |
| 01 | Agent Loop | query.ts TAOR、AsyncGenerator | [cc-agent-loop.md](../notes/cc-agent-loop.md) |
| 02 | System Prompt | 静/动态区、cache hit | — |
| 03 | Tool System | 40+ tools、BashTool | cc-agent-loop |
| 04 | Commands | 80+ slash commands | — |
| 05 | Context Management | compaction 流水线 | — |
| 06 | Permission & Security | 17K LOC、deny-first | [cc-permission-model.md](../notes/cc-permission-model.md) |
| 07 | Multi-Agent | fork/coordinator | — |
| 08 | MCP & Services | 8 transports | [cc-mcp-hooks.md](../notes/cc-mcp-hooks.md) |

## 综合综述

### [pankaj28843/understanding-claude-code](https://github.com/pankaj28843/understanding-claude-code)
- **type**: analysis | **priority**: P1 | **status**: todo
- **notes**: 26 篇文章 + HN 讨论合成

## 深度文章

### [512K 行代码深度解构（中文）](http://www.diginfo.me/claude-code-source-deep-dive)
- **type**: analysis | **priority**: P1 | **status**: todo
- **notes**: Feature Gate、TAOR、权限双引擎

### [Architecture not Features（Medium）](https://medium.com/data-science-collective/everyone-analyzed-claude-codes-features-nobody-analyzed-its-architecture-1173470ab622)
- **type**: analysis | **priority**: P1 | **status**: todo
- **notes**: AsyncGenerator 自愈循环、编译时 Feature Elimination
