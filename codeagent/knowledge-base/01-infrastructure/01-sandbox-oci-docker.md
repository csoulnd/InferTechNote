# 底层隔离 & OCI / Docker / crun / kata / bwrap

> 介绍级文档：建立原理认知与场景映射，深入实践见参考链接。

## 原理

Linux 通过 **Namespace**（进程、网络、挂载、UTS、IPC、用户、cgroup 等视图隔离）和 **cgroup**（CPU/内存/IO/PID 配额）实现进程级隔离，这是所有容器技术的内核基础。

**OCI（Open Container Initiative）** 定义了两份核心规范：**Runtime Spec**（容器如何运行）和 **Image Spec**（镜像如何打包）。**runc** 是 OCI 参考实现；**crun** 是用 C 编写的轻量 runtime；**Kata Containers** 在轻量 VM 中跑容器，防内核逃逸；**bubblewrap（bwrap）** 常用于桌面/本地最小沙箱。

**Docker** 在 OCI runtime 之上提供镜像构建、网络、存储、编排等工程化能力。容器运行中的 bind mount、`docker cp` 等操作底层依赖 mount namespace；**nsenter** 可进入已有 namespace 做动态挂载调试。

## 典型应用场景

- **三方 CodeAgent 容器化**：Docker 拉起预装 CC/OC + sshd 的镜像，Gateway 经 SSH 连接容器
- **GPU 推理容器**：在容器中部署 vLLM，挂载 GPU 设备与模型卷
- **轻量本地沙箱**：jiuwenbox 用 bubblewrap + namespace + Landlock + seccomp，不依赖完整 Docker daemon
- **安全分级**：普通多租户用 Docker；高敏感 workload 可选 Kata 增强隔离

## 参考链接

### 官方标准

- [namespaces(7)](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [cgroups(7)](https://man7.org/linux/man-pages/man7/cgroups.7.html)
- [OCI 官网](https://opencontainers.org/)
- [OCI Runtime Spec](https://github.com/opencontainers/runtime-spec)
- [OCI Image Spec](https://github.com/opencontainers/image-spec)
- [runc](https://github.com/opencontainers/runc)
- [crun](https://github.com/containers/crun)
- [Kata Containers](https://katacontainers.io/docs/latest/)
- [bubblewrap](https://gitlab.com/bubblewrap/bubblewrap)
- [Docker Get Started](https://docs.docker.com/get-started/)
- [Docker Bind Mounts](https://docs.docker.com/storage/bind-mounts/)
- [nsenter(1)](https://man7.org/linux/man-pages/man1/nsenter.1.html)

### 教程 / 课程

- [阿里云云原生 AI 容器实验（含 GPU 容器部署 LLM）](https://developer.aliyun.com/adc/scenarioSeries/e5427732f6e94cde939a7aeed1d19180)

### 开源项目

- [opencontainers/runc](https://github.com/opencontainers/runc) — 最小 OCI runtime 实验

### agentos 对照

- `agentos/jiuwenswarm/jiuwenbox/` — bubblewrap 沙箱、namespace/cgroup/Landlock policy
