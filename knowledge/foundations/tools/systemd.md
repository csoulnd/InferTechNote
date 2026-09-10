---
title: "systemd：Linux 系统与服务管理器"
type: concept
domain: foundations
status: active
---

# systemd：Linux 系统与服务管理器

## 一句话解释

systemd 是 Linux 的系统与服务管理器，以 unit（单元）描述服务等资源，管理启动、停止、依赖与进程生命周期。

## 工作原理

系统级 systemd 通常作为 PID 1 运行；也有独立的用户级管理器。unit 不只有 `.service`，还包括 `.socket`、`.timer`、`.mount` 和 `.target` 等类型。`systemctl` 是操作管理器的命令行工具。

服务单元声明运行命令、用户、工作目录和重启策略，systemd 通过 cgroup 跟踪服务进程。下面是系统服务示意，假设 `app` 用户和程序路径已存在，程序以前台方式运行：

```ini
# /etc/systemd/system/example.service
[Unit]
Description=Example worker

[Service]
Type=exec
User=app
WorkingDirectory=/opt/example
ExecStart=/opt/example/bin/worker
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

`Type=exec` 等待程序成功执行，但不等于业务已就绪。`Restart=on-failure` 在符合条件的失败后重启，正常退出和显式停止不会因此重启；重启还受启动频率限制。具体选项支持情况以目标机器的 systemd 版本为准。

## 实践意义

```bash
sudo systemctl daemon-reload          # 修改单元后重新加载定义
sudo systemctl enable --now example   # 配置开机启动，同时立即启动
systemctl status example             # 查看状态
journalctl -u example -n 50           # 查看最近日志
sudo systemctl stop example          # 停止服务
```

`enable` 本身不立即启动，`start` 本身不配置开机启动。修改应用代码或配置后，`daemon-reload` 也不会替你重启应用。服务默认可通过 journal 收集标准输出和错误，实际行为受日志配置影响。

## 适用边界

- 适合主机上的长期服务；`ExecStart` 默认不经过 shell，不能直接按 shell 语法使用管道、重定向或 `&`。
- `After=` 只规定顺序，不会单独拉起依赖，也不保证远端接口可用；应用仍需处理依赖暂时不可达。
- 自动重启不等于业务健康检查，也不能解决整台机器宕机；跨机器容错需额外的高可用设计。

## 相关知识

- [nohup：忽略挂断信号运行命令](nohup.md)
- [HA：高可用](../reliability/high-availability.md)

## 参考资料

- [systemd 官方手册](https://www.freedesktop.org/software/systemd/man/latest/systemd.html)
- [systemd.service 官方手册源文件](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml)
- [systemctl 官方手册](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html)
