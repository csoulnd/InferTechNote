---
title: "whl：Python Wheel 分发格式"
type: concept
domain: foundations
status: active
---

# whl：Python Wheel 分发格式

## 一句话解释

`.whl` 是 Python Wheel 文件的扩展名，表示已经构建好的 Python 分发包，安装工具可将其中的代码、资源和元数据安装到兼容的 Python 环境。

## 工作原理

Wheel 本质上是采用规定布局的 ZIP 归档，包含待安装文件和 `.dist-info` 元数据目录。与需要先构建的源码分发包 sdist 相比，安装已有 wheel 无需再次运行该包的构建过程。

典型文件名为：

```text
example-1.0.0-py3-none-any.whl
│       │     │   │    └─ 平台标签：任意平台
│       │     │   └────── ABI 标签：无特定 ABI 要求
│       │     └────────── Python 标签：Python 3
│       └──────────────── 版本
└──────────────────────── 分发名称
```

完整格式允许在版本后加入可选的构建标签。含原生扩展的 wheel 通常带有更具体的 Python、ABI 和平台标签，安装器据此判断兼容性；还需满足包元数据中的 Python 版本等要求。

```bash
python -m pip install ./example-1.0.0-py3-none-any.whl
```

示例假设该文件已存在。发布链路通常为：`Python 项目 → 构建后端生成 wheel → 上传 PyPI 或私有索引 → pip 安装`。

## 适用边界

- “已构建”不意味着一定包含机器码：纯 Python 代码也能打成 wheel。
- wheel 是文件格式，pip 是安装工具，PyPI 是包索引服务，三者不是同一层次。
- 一个 wheel 通常不包含全部依赖；离线安装还需要准备兼容的依赖包。
- wheel 主要面向 Python 环境，不负责像系统包一样统一管理系统服务和操作系统依赖。

## 实践意义

选择或发布 wheel 时，应确认 Python 版本、操作系统、CPU 架构和 ABI 兼容性；不能只看包名和版本。原生扩展可能需要分别构建多个平台的 wheel。

## 相关知识

- [npm：JavaScript 包管理与发布生态](npm.md)
- [deb：Debian 二进制软件包格式](deb.md)

## 参考资料

- [PyPA：Binary distribution format（Wheel）](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)
- [PyPA：Platform compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)
