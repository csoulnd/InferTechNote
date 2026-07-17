# 容器镜像的容器化构建

## 使用docker构建容器镜像

### 构建流程

挂载宿主机docker和docker.sock，并且配置容器用户的组为docker。不需要其他特权，支持普通用户。

docker容器镜像
```Dockerfile
FROM openeuler/openeuler:24.03

ARG USR_NAME=testuser
ARG USR_HOME=/home/testuser
ARG DOCKER_GROUP=107#通过命令getent group docker查看

RUN useradd $USR_NAME && groupadd -g $DOCKER_GROUP docker && usermod -a -G docker $USR_NAME && mkdir -p $USR_HOME && chown -R $USR_NAME:$USR_NAME $USR_HOME

WORKDIR $USR_HOME
USER $USR_NAME:$DOCKER_GROUP
```

使用如下命令测试
```bash
# test.Dockerfile，用于在容器中构建镜像
cat > test.Dockerfile <<EOF
FROM openeuler/openeuler:24.03

COPY test.txt .

CMD ["/bin/bash", "-c", "cat test.txt"]
EOF

# test.txt，测试COPY指令
echo "123456" > test.txt

# 构建镜像，需要提前下载openeuler.tar.gz到当前目录；docker:1.0是前述Dockerfile构建的镜像
docker run --rm -it -v /usr/bin/docker:/usr/bin/docker -v /run/docker.sock:/run/docker.sock -v $(pwd):/home/testuser/work -v /tmp:/tmp docker:1.0 /bin/bash -c "\
  cd work &&\
  docker load -i openeuler.tar.gz &&\
  docker build -t test:1.0 -f test.Dockerfile . &&\
  docker save test:1.0 | gzip -c > /tmp/test.tar.gz"

# 运行新构建的镜像
docker load -i /tmp/test.tar.gz && docker run --rm -it test:1.0
```

耗时 11.07（没有计入openeuler镜像的加载时间） 45.1kB (virtual 459MB)

### 外部依赖

宿主机：docker
容器：配置容器用户加入docker组

### 权限风险分析

#### docker.sock

作用：用于控制宿主机docker。

风险：攻击者若控制构建容器，攻击者直接通过docker创建特权容器，实现逃逸（[案例](https://www.freebuf.com/articles/system/382166.html)）。

## 使用buildah构建容器镜像

buildah无需守护进程，也不需要配置特殊用户。buildah支持vfs和overlay两种存储引擎：vfs配置简单，但是耗时和占用空间较大；overlay需要一定的配置，速度较快且占用空间较小。

### 基于vfs存储引擎的buildah构建镜像

buildah容器镜像
```Dockerfile
FROM openeuler/openeuler:24.03

ARG USR_NAME=testuser
ARG USR_HOME=/home/testuser

RUN yum install -y buildah

RUN useradd $USR_NAME && mkdir -p $USR_HOME && chown -R $USR_NAME:$USR_NAME $USR_HOME

WORKDIR $USR_HOME
USER $USR_NAME:$USR_NAME
```

使用如下命令测试
```bash
# 构建镜像，需要提前下载openeuler.tar.gz到当前目录；buildah:1.0是前述Dockerfile构建的镜像
docker run --rm -it --security-opt apparmor=unconfined --security-opt seccomp=unconfined --security-opt systempaths=unconfined --cap-add SYS_ADMIN -v $(pwd):/home/testuser/work -v /tmp:/tmp buildah:1.0 /bin/bash -c "\
  cd work &&\
  buildah pull --storage-driver=vfs docker-archive:openeuler.tar.gz &&\
  buildah bud --storage-driver=vfs -t test:1.0 -f test.Dockerfile . &&\
  buildah push --storage-driver=vfs test:1.0 docker-archive:/tmp/test.tar.gz"

# 运行新构建的镜像，这里镜像增加hostname localhost
docker load -i /tmp/test.tar.gz && docker run --rm -it localhost/test:1.0
```

### 基于overlay存储引擎的buildah构建镜像

构建命令如下，与vfs存储引擎的区别是需要挂载/dev/fuse，并且需要将storage-driver参数改为overlay
```bash
# 构建镜像，需要提前下载openeuler.tar.gz到当前目录
docker run --rm -it --security-opt apparmor=unconfined --security-opt seccomp=unconfined --security-opt systempaths=unconfined --cap-add SYS_ADMIN --device /dev/fuse -v $(pwd):/home/testuser/work -v /tmp:/tmp buildah:1.0 /bin/bash -c "\
  cd work &&\
  buildah pull --storage-driver=overlay docker-archive:openeuler.tar.gz &&\
  buildah bud --storage-driver=overlay -t test:1.0 -f test.Dockerfile . &&\
  buildah push --storage-driver=overlay test:1.0 docker-archive:/tmp/test.tar.gz"
```

### 性能对比

vfs存储引擎 22.3s 573MB (virtual 1.03GB)

overlay存储引擎 18.3s 192MB (virtual 650MB)

### 外部依赖

宿主机：docker，使用overlay存储引擎时，要求宿主机内核版本>=5.4。
容器：buildah

### 权限风险分析

#### CAP_SYS_ADMIN

作用：挂载文件系统和创建隔离的命名空间，[issue#4563](https://github.com/podman-container-tools/buildah/issues/4563)。

风险：CAP_SYS_ADMIN权限允许容器执行部分高危系统调用，如挂载文件系统，加载内核模块等。

#### 放开secomp过滤

作用：使用mount，unshare等系统调用。

风险：mount等高风险系统调用能够执行，降低容器逃逸的难度。

#### 放开apparmor权限控制

作用：支持容器内挂载overlayfs。

风险：容器内进程能够访问/proc，/sys等特殊文件，降低容器逃逸的难度。

#### 关闭landlock系统路径限制（--security-opt systempaths=unconfined）

作用：容器内进程可以访问所有虚拟文件系统，如/proc等，这些文件在构建时需要被访问

风险：容器内进程能够访问/proc，/sys等特殊文件，降低容器逃逸的难度。

#### 挂载/dev/fuse设备

作用：支持overlayfs，提高构建速度，减少磁盘占用。

风险：用户态overlayfs文件系统，风险较低。需要配合CAP_SYS_ADMIN、apparmor、seccomp等漏洞或不安全配置实现逃逸。

## 使用buildkit构建容器镜像

buildkit不需要使用特殊用户，也不需要特权容器。buildkit编译镜像只需要配置apparmor和seccomp，不需要配置CAP_SYS_ADMIN。

### 外部依赖

宿主机：
- 配置sysctl.conf，`kernel.apparmor_restrict_unprivileged_userns=0`
容器：
- fuse3
- shadow-utils
- 配置subuid/subgid

### 权限风险分析

buildkit的容器化构建需要apparmor，secomp和landlock三类特权。具体风险已在上文分析。

# 容器化构建的风险缓解措施

缓解措施：构建容器独立部署，与控制面独立，严格校验对外参数。禁止直接运行前端上传npm包或者二进制。

构建容器对外接口：

```bash
def build(
    task_id: str,
    agent_id: str,
    version: str,
    tgz_path: Path,
    output_dir: Path): ...
```

接口通过unix socket或https协议对外暴露，并使用非对称加密密钥或配置文件权限进行访问控制。
