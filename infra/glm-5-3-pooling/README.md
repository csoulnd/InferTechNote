---
title: "GLM-5.3 双机池化部署与测试"
type: moc
domain: infra
status: active
---

# GLM-5.3 双机池化部署与测试

- [部署指南](deployment.md)：PCP8/DP8、容器挂载、UB rootinfo与topo、MMC配置、基线及池化启动。
- [测试方法](test-plan.md)：通用case、原生AISBench数据/预热/发压、配对比较及介质证据边界；不含安装。
- [无池化脚本](scripts/baseline/launch.sh)、[池化脚本](scripts/pcp-pool/launch.sh)：参数以脚本为准。
- [工作区技能](../../.agents/skills/service-test-workspace/SKILL.md)：服务日志按启动批次，测试原生结果按case/轮次保存。
- [返回基础设施索引](../README.md)。

服务器工程为/home/d00937186/glm53-pooling-lab，外层MMC配置与各机rootinfo是唯一来源。运行日志、权重及本地归档不提交到仓库。旧poolbench和TP8实验材料不是当前手动测试入口。
