---
title: "Debian 与 deb：发行版及其软件包格式"
type: concept
domain: foundations
status: evergreen
---

# Debian 与 deb：发行版及其软件包格式

## 一句话解释

Debian 是社区驱动的操作系统项目，`.deb` 是 Debian 及 Ubuntu 等衍生发行版使用的二进制软件包格式，用于按系统包管理规则安装文件、声明依赖并管理软件生命周期。

## Debian、APT 与 deb 的关系

Linux 内核负责进程、内存、设备和网络等底层机制；Debian GNU/Linux 发行版进一步集成用户空间工具、库、安装程序和软件仓库，形成可安装、更新和维护的操作系统。Debian 常见发布分支包括 stable、testing 和 unstable。

在软件发布语境中，可以按下面的关系理解：

```text
Debian：目标操作系统发行版
  └─ APT 仓库：组织软件包、索引及更新的分发渠道
       └─ .deb：实际安装的软件包文件
```

系统使用 dpkg 管理本地包，使用 APT 从配置的软件源解析依赖、获取软件和更新。

## deb 的结构与安装

deb 文件是 ar 归档，主要包含格式版本文件 `debian-binary`、控制信息归档 `control.tar.*` 和待安装数据归档 `data.tar.*`。控制信息包括包名、版本、架构和依赖，也可以包含安装、升级或卸载期间执行的维护脚本。

常见文件名为 `example_1.0-1_amd64.deb`，分别体现包名、版本和架构；包管理器以包内元数据为准，而不是仅依赖文件名。

- **dpkg**：负责本地包的解包、配置及安装状态管理；不会自动从仓库下载缺失依赖。
- **APT**：结合仓库索引解析依赖、获取包，并调用底层工具完成安装。

```bash
dpkg-deb --info ./example_1.0-1_amd64.deb      # 查看包元数据
dpkg-deb --contents ./example_1.0-1_amd64.deb  # 查看包内文件
sudo apt install ./example_1.0-1_amd64.deb    # 安装本地包并尝试解析依赖
```

示例假设文件已存在，且使用支持本地 deb 安装的 APT；缺失依赖仍需能从已配置软件源获得。

## 适用边界

- deb 是包格式，Debian 是发行版；Ubuntu 使用 deb，不代表任意 Debian 包都能在 Ubuntu 上兼容运行。
- Debian 不等于 Linux 内核；“支持 Debian”仍需说明发行版版本与 CPU 架构。
- stable 是 Debian 的发布分支及维护取向，不表示软件永远不变或绝无故障。
- “二进制包”表示可供安装的构建结果，也可以只包含脚本或数据，并非一定含机器码。
- deb 可以安装命令、库、配置和 systemd 单元等；实际是否启用或启动服务取决于包脚本和系统策略。
- 制作 deb 与建立 APT 仓库是两个步骤：仓库还需要索引及相应的信任配置，单独上传 deb 不等于完成仓库发布。

## 实践意义

发布系统级软件时，需要明确目标发行版及版本、架构、依赖来源和升级行为。语言包和 deb 可以配合使用，例如将 Python 应用连同系统集成文件封装成 deb；已有 npm 包或 wheel，并不代表已经完成 Debian 系统集成。

## 相关知识

- [Linux 快速入门：核心认知与学习路线](../linux/README.md)
- [systemd：Linux 系统与服务管理器](systemd.md)
- [whl：Python Wheel 分发格式](wheel.md)

## 参考资料

- [Debian FAQ：包管理基础](https://www.debian.org/doc/manuals/debian-faq/pkg-basics.en.html)
- [Debian 手册：deb 文件格式](https://manpages.debian.org/deb.5)
- [Debian 官方：About Debian](https://www.debian.org/intro/about)
- [Debian 官方：发行版本](https://www.debian.org/releases/)
