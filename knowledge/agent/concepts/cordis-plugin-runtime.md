---
title: "Cordis 插件运行时"
type: concept
domain: agent
status: active
---

# Cordis 插件运行时

## 核心问题

Cordis 是什么，它如何让多个插件组合成一个可运行、可替换、可卸载的应用？

## 一句话解释

Cordis 是一个**负责组装和管理插件的运行时框架**。

你可以先把它想成一块智能插线板：

- 每个电器是一个插件。
- 插座上的能力是 Service。
- 插件需要哪种能力，就通过 `inject` 声明。
- 插线板负责等依赖就绪后再接通插件。
- 拔掉插件时，插线板还负责撤销监听器、计时器等资源。

这个比喻只能帮助入门。Cordis 实际管理的是软件插件、依赖关系、事件和生命周期，而不是电源。

## 为什么需要 Cordis

假设一个 Agent 应用包含模型、工具、会话存储和 Web UI。最直接的写法是让它们互相导入并手动初始化：

```text
main
├─ new ModelClient()
├─ new ToolRegistry(modelClient)
├─ new SessionStore()
└─ new WebUI(toolRegistry, sessionStore)
```

这种方式在组件较少时很直观，但组件增加后会遇到几个问题：

1. 初始化顺序写死在 `main` 中。
2. 消费者容易依赖某个具体实现，替换模型或存储会牵动很多代码。
3. 热更新或关闭时，需要手动追踪每个监听器、连接和计时器。
4. 扩展功能往往必须修改主程序。

Cordis 把这些问题转换成“插件提供什么、需要什么、何时清理”的声明式装配问题。

## 最小模型：先掌握四个概念

初学时只需要理解 Plugin、Context、Service 和 `inject`。Event、Effect、Fiber 可以稍后学习。

### 1. Plugin：一个功能模块

Plugin 是 Cordis 的最小装配单元。最简单的插件只是一个接收 `ctx` 的函数：

```ts
import type { Context } from '@deepseek-ai/cordis'

export function apply(ctx: Context) {
  console.log('plugin started')
}
```

Cordis 加载插件时调用 `apply()`，卸载插件时撤销该插件作用域内登记的资源。

### 2. Context：插件与运行时交互的入口

`Context` 通常写作 `ctx`。插件通过它：

- 获取别的插件提供的能力，例如 `ctx.tools`。
- 提供自己的 Service。
- 监听或触发事件。
- 注册需要随插件卸载的资源。

它既像一个“能力目录”，也代表当前插件的生命周期作用域。不要把它简单理解成装有任意变量的全局对象。

### 3. Service：插件提供给其他插件的能力

Service 是一个有名字、有接口的能力。例如：

```text
tools    → 注册和执行工具
llm      → 调用模型
sessions → 创建和读取会话
```

消费者依赖的是 Service 契约，而不是具体实现：

```text
Tool Consumer → tools 接口 ← Local Tool Provider
                         ← Remote Tool Provider
```

只要两个 Provider 实现同一个契约，就可以替换，而 Consumer 无须知道背后的类名或包名。

### 4. `inject`：声明插件需要哪些 Service

如果插件需要工具服务，可以声明：

```ts
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(/* tool */)
}
```

含义是：“只有 `tools` Service 可用时，才启动我。”

因此配置文件中的上下顺序不等于启动顺序。Cordis 根据实际依赖决定何时激活插件；依赖消失或被替换时，相关消费者也可能卸载并重新启动。

## 一个完整的小例子

设想两个插件：一个提供问候能力，一个消费它。

```ts
// greeter-provider.ts
import { Service, type Context } from '@deepseek-ai/cordis'

export class Greeter extends Service {
  constructor(ctx: Context) {
    super(ctx, 'greeter')
  }

  hello(name: string) {
    return `Hello, ${name}`
  }
}
```

```ts
// greeter-consumer.ts
import type { Context } from '@deepseek-ai/cordis'

export const inject = ['greeter']

export function apply(ctx: Context) {
  console.log(ctx.greeter.hello('DSH'))
}
```

运行时关系是：

```text
Greeter Provider ──提供──> greeter Service
                              │
                              └──注入──> Greeter Consumer
```

如果换成中文问候 Provider，只要它仍提供相同的 `greeter` 契约，Consumer 就不需要修改。

## 第二阶段：事件如何让插件协作

Service 适合“调用一个明确能力”，Event 适合“宣布发生了一件事”。

### Event：广播通知

例如会话追加了一条消息后广播 `session/event`，日志、UI 和遥测插件都可以独立监听：

```text
session/event
├─ persistence listener
├─ UI listener
└─ telemetry listener
```

发送方不需要逐个导入这些消费者。

### Waterfall：可按顺序处理的决策链

Waterfall 会把一个值依次交给处理者。每个处理者可以继续传递、修改结果，或按协议停止后续处理。它适合权限判断、请求包装或工具执行流水线。

简单区分：

- “某件事已经发生，请大家知道”——Event。
- “请多个插件共同决定或变换结果”——Waterfall。
- “请某个能力完成明确操作”——Service。

## 第三阶段：为什么卸载不会到处泄漏

插件通常会创建副作用，例如事件监听器、计时器、连接和子进程。如果插件卸载后这些资源仍存活，就会产生重复器重复触发、使用旧配置或进程泄漏。

Cordis 用 Effect 和 Fiber 管理生命周期：

- **Effect**：一次带清理动作的注册或资源创建。
- **Fiber**：承载一组相关 Effect 的生命周期作用域。
- **disposer**：撤销资源的函数。

例如：

```ts
export function apply(ctx: Context) {
  ctx.effect(() => {
    const timer = setInterval(() => console.log('tick'), 5_000)
    return () => clearInterval(timer)
  })
}
```

插件卸载时，Cordis 调用返回的 disposer。可把它理解为：插件不仅声明“启动时做什么”，也把“退出时怎么收尾”登记给运行时。

注意，自动清理不是魔法。只有通过 Cordis 注册或显式提供 disposer 的资源才能被正确追踪；插件自己偷偷创建又不登记的资源仍会泄漏。

## Loader 与配置树

Loader 读取配置并把插件挂载成一棵树。配置描述“要装哪些插件以及各自配置”，依赖则描述“它们何时能够运行”。

```text
配置文件
  → Loader 解析插件条目
  → 创建插件作用域
  → 等待 inject 依赖
  → 调用 apply(ctx, config)
  → 记录 Service、Event 和 Effect
  → 配置变化/依赖消失/关闭
  → dispose 对应作用域
```

在 DSH 中，profile、bundle 和 patch 先合成最终 Cordis 配置树，Loader 再挂载它。它们是 DSH 的发行与配置机制，不是理解 Cordis 本身的前置条件。

## 把 Cordis 映射回 DSH

现在再看 DSH 的“一切皆插件”：

| Cordis 概念 | DSH 中的例子 |
|---|---|
| Plugin | 模型适配器、工具、会话持久化、Web UI、Agent Loop |
| Context | 插件访问 `ctx.llm`、`ctx.tools`、`ctx.sessions` 的入口 |
| Service | LLM、Tool、Session、Filesystem、Sandbox 等能力契约 |
| `inject` | Tool 插件等待 `tools` Service，Consumer 等待 Provider |
| Event | `session/event`、`agent/*` 生命周期通知 |
| Waterfall | 模型请求或工具执行的拦截与变换链 |
| Effect/Fiber | 插件热更新、依赖替换和关闭时的资源清理 |
| Loader | 根据最终配置树挂载整个 Harness |

所以，DSH 不是“先写好一个固定 Agent，再外挂几个插件”，而是“通过插件和 Service 组合出这个 Agent”。

## 常见误解

### Cordis 是依赖注入容器吗？

依赖注入是它的一部分，但不完整。Cordis 还管理插件树、事件、作用域和可逆生命周期。

### Cordis 是消息队列吗？

不是。Event 能解耦生产者与消费者，但 Cordis 不等于跨进程消息中间件，也不默认提供持久队列语义。

### Cordis 的 Context 是全局变量吗？

不是简单的全局变量。Context 既提供能力视图，也携带作用域；不同子上下文可以看到不同注册项，并拥有不同的清理边界。

### 有了 `inject`，YAML 顺序就完全没意义吗？

Service 启动依赖不靠列表顺序，但配置组合和 patch 层仍有顺序与覆盖语义。不要混淆“配置覆盖顺序”和“插件依赖启动顺序”。

### 自动卸载就等于安全隔离吗？

不是。生命周期管理避免资源泄漏，但插件仍可能与宿主共享进程、文件、网络和凭据权限。

## 适用边界

- 本文解释 Cordis 的通用运行时模型，并用 DSH 举例；不覆盖每个 DSH package 的业务 API。
- DSH 与 Cordis 仍在快速演进，具体 API、包名和配置格式应以锁定版本的类型与官方文档为准。
- 动态插件、自修改 Runtime 和浏览器侧插件属于更高风险扩展面，需要另外设计审批、审计和回滚。

## 学习检查

如果可以回答下面五个问题，就具备继续阅读 DSH 架构的基础：

1. Plugin、Service 和 Provider 分别是什么？
2. `ctx` 为什么不只是普通全局对象？
3. `inject` 解决什么问题？
4. Event 和 Service 应该如何选择？
5. 一个插件创建计时器或连接后，卸载时如何清理？

## 实践意义

- 插件应依赖稳定的 Service 契约，而非具体 Provider 类。
- 用 `inject` 表达启动条件，不用配置位置暗示依赖。
- 所有监听器、连接、Worker、进程和计时器都应绑定作用域或 disposer。
- 排障顺序应是：配置树是否包含插件 → 模块能否解析 → schema 是否通过 → 依赖是否齐备 → `apply()` 是否失败。
- Provider 替换、依赖消失和 reload/unload 都应有集成测试。

## 应用记录

- [DeepSeek Harness 学习路径](../../../agent/study-notes/dsh/README.md)
- [DeepSeek Harness 插件系统学习报告](../../../agent/study-notes/dsh/plugin-system.md)
- [OpenCode 插件系统学习报告](../../../agent/study-notes/opencode/plugin-system.md)

## 相关知识

- [Hook 扩展机制](hook-mechanism.md)
- [Agent 架构总览](../architecture/overview.md)
- [Sandbox 生命周期](../integration/sandbox-lifecycle.md)

## 参考资料

- [Cordis Primer](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.zh.md)
- [DeepSeek Harness Cordis Tutorial](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-tutorial/index.zh.md)
- [Cordis Registry API](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-api/registry.md)
- [DeepSeek Harness Architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.zh.md)
