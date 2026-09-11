---
title: "npm：JavaScript 包管理与发布生态"
type: concept
domain: foundations
status: active
---

# npm：JavaScript 包管理与发布生态

## 一句话解释

npm 是 JavaScript 生态中的包管理与发布体系，包括命令行工具 npm CLI、存储包的 registry（包仓库服务）和 npm 网站；日常说“运行 npm”通常指命令行工具。

## 工作原理

一个 npm 包通常通过 `package.json` 声明名称、版本、依赖和入口等信息。发布者把包内容打包并上传到 registry，使用者通过包名和版本安装，包管理器解析依赖并把所需内容安装到项目中。

发布流程可理解为：`源码与构建产物 → npm pack 生成 .tgz → npm publish 发布到 registry → npm install 安装`。`npm publish` 可以直接从包目录发布，不要求先手动执行 `npm pack`。

```bash
npm pack             # 在当前包目录生成可分发的 .tgz 文件
npm install ./example-1.0.0.tgz  # 安装一个已有的本地包
```

示例文件名仅为说明。发布前应确认包内包含运行所需产物；打包并不自动保证源码已经编译。

## 适用边界

- npm 是工具与服务体系，`.tgz` 才是常见的包文件形式；不存在与 `.whl`、`.deb` 对等的通用 `.npm` 扩展名。
- Node.js 是 JavaScript 运行时，npm 是包管理工具；前端库、构建工具和命令行应用也可以通过 npm 分发。
- `npm publish` 发布软件包，不等于把网站或后端服务部署上线。
- registry 可以是公共 npm registry，也可以是兼容的私有服务；安装内容及生命周期脚本行为受包内容、配置和 CLI 版本影响。

## 实践意义

讨论“发布 npm”时，应明确包名、版本、目标 registry 和要包含的文件。`package.json` 中的依赖范围与 `package-lock.json` 记录的具体依赖解析结果用途不同。

## 相关知识

- [whl：Python Wheel 分发格式](wheel.md)
- [deb：Debian 二进制软件包格式](deb.md)

## 参考资料

- [npm 官方：About npm](https://docs.npmjs.com/about-npm/)
- [npm pack](https://docs.npmjs.com/cli/commands/npm-pack)
- [npm publish](https://docs.npmjs.com/cli/commands/npm-publish)
