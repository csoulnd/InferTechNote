---
title: "nohup：忽略挂断信号运行命令"
type: concept
domain: foundations
status: evergreen
---

# nohup：忽略挂断信号运行命令

## 一句话解释

nohup（no hangup）让启动的命令忽略 SIGHUP（挂断信号），常用于让临时任务在终端或 SSH 会话断开后继续运行。

## 工作原理

nohup 设置忽略 SIGHUP 后执行命令；它不会持续监督进程，也不会自动把任务放入后台。后台执行由 shell 的 `&` 完成。

```bash
nohup python3 job.py > job.log 2>&1 < /dev/null &
```

- `> job.log`：将标准输出写入文件；会截断已有文件，追加可用 `>>`。
- `2>&1`：让标准错误使用当前标准输出的去向，因此放在输出重定向之后。
- `< /dev/null`：显式断开终端输入；需要交互输入的程序不适合这个示例。
- `&`：让 shell 在后台运行命令；仅用 `&` 不代表忽略 SIGHUP。

GNU nohup 在标准输出仍指向终端时，通常追加到当前目录的 `nohup.out`，失败时尝试 `$HOME/nohup.out`；标准错误仍指向终端时也会调整其去向。显式重定向更便于确定日志位置。

## 适用边界

- 只处理挂断信号这一类问题，不能防止程序崩溃、OOM、机器重启或其他终止信号。
- 会话清理策略可能直接终止进程；程序也可能重新设置 SIGHUP 处理方式，因此不能保证所有环境下退出登录后都存活。
- 不提供失败重启、开机启动、健康检查和日志轮转；nohup.out 也不证明任务成功完成。
- 不同系统实现的默认输入输出处理可能不同，本文默认行为以 GNU Coreutils 为例。

## 实践意义

临时批处理可用 nohup，并检查日志与任务产物；长期服务需要统一启停和失败恢复时，使用服务管理器更合适。

## 相关知识

- [systemd：Linux 系统与服务管理器](systemd.md)
- [HA：高可用](../reliability/high-availability.md)

## 参考资料

- [GNU Coreutils：nohup invocation](https://www.gnu.org/software/coreutils/manual/html_node/nohup-invocation.html)
