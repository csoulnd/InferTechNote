# Claude Code 官方用法

## P0 必读

### [Claude Code Setup](https://code.claude.com/docs/en/setup)
- **type**: standard | **priority**: P0 | **status**: todo
- **notes**: 安装、环境变量、认证方式

### [How Claude Code Works](https://code.claude.com/docs/en/how-claude-code-works)
- **type**: standard | **priority**: P0 | **status**: todo
- **notes**: Agentic Loop：gather context → take action → verify results；agentic harness 概念

### [Agent SDK / Headless](https://code.claude.com/docs/en/how-claude-code-works)
- **type**: standard | **priority**: P0 | **status**: todo
- **notes**: `--print` 无 TUI 模式，对应 Gateway 消息路径 `claude --print`

### [MCP Glossary](https://code.claude.com/docs/en/glossary)
- **type**: standard | **priority**: P0 | **status**: todo
- **notes**: MCP 连接外部工具；容器镜像需预配 MCP server

### [Hooks](https://code.claude.com/docs/en/hooks)
- **type**: standard | **priority**: P1 | **status**: todo
- **notes**: UserPromptSubmit 等事件钩子；对照 Gateway Hook 设计

## 本地验收

- [ ] 安装并跑通 `claude` TUI
- [ ] 跑通 `claude --print "hello"` headless 模式
