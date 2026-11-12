---
title: "GLM-5.3 池化基线与80/20性能测试计划"
type: design
domain: infra
status: draft
---

# GLM-5.3 池化基线与80/20性能测试计划

配套 [部署教程](deployment.md) 与 [脚本说明](README.md)。本文同时记录测试方法、已验证证据和剩余验收项。§3/§4指部署教程；§5.x指本文对应步骤。

## 5. 后续测试计划与操作指南

> 目标：1P1D，输入128K / 输出1K，多前缀、共享前缀比例90%；排除HBM前缀复用，比较无池化基线与DRAM/SSD读取比例80/20时的吞吐。已完成的功能诊断不等于正式性能验收，当前进展见§5.0.1。

### 测试目标与方法对应

| 目标 | 方法 | 验收边界 |
| --- | --- | --- |
| 排除HBM复用 | 两组使用--no-enable-prefix-caching，核对进程参数与本地命中差值 | 不使用skip_reading_prefix_cache；不以响应加速判断命中 |
| 验证池化读写 | 首次写入、等待完成、重复前缀读取，关联P端命中与Meta读写指标 | 本地复用0，外部读取为正，请求成功 |
| 确保128K/1K与90%共享 | 固定生成器版本、seed、数据哈希和顺序，核验实际usage和完整块前缀 | 共享前缀约90%；独有后缀不能被整请求预热 |
| 标定容量与并发 | 1/2/4独立前缀测占用；并发1、2检查OOM、抢占和队列 | 当前仅验证到并发2，不以服务上限16代表128K安全并发 |
| 证明物理SSD读取 | 淘汰后冷读，关联回温对象/字节、P外部命中、两台专用分区读写差值 | 区分UBS IO缓冲，物理I/O仅作交叉验证 |
| 验证80/20 | 每轮重建热冷布局，以外部读取KV字节计算比例 | SSD目标20%±2个百分点；串行诊断不替代并发2验收 |
| 公平比较收益 | 基线移除外部池连接器、保留PD；同数据/顺序/MTP/并发，各至少3轮 | 排除预热时间；报告中位数、波动及TTFT/TPOT，区分端到端与P阶段 |

### 5.0 测试顺序与通过条件

| 阶段 | 操作 | 通过条件 |
| --- | --- | --- |
| 1 | 准备独立AISBench环境及固定数据 | CLI、tokenizer和小请求可用；记录版本与数据哈希 |
| 2 | 关闭本地前缀缓存，验证外部池 | HBM无复用，池写入及重复请求读取成功 |
| 3 | 128K单并发试跑、KV容量标定 | 无OOM，长度符合要求，得到每前缀KV容量 |
| 4 | 构造热/冷前缀并采集分介质证据 | 可区分L2、UBS IO缓冲和物理SSD读取 |
| 5 | 同条件测基线和池化组 | 同数据/顺序/并发，完成至少3轮有效测试 |
| 6 | 汇总与收尾 | 保留日志、配置、指标；按约定保留或停止本任务服务 |

本轮已验证关闭L1后的短请求外部池读写及128K/1K并发1、2，详见§5.0.1；稳定80/20及正式基线对比仍待验证。69容器已准备独立AISBench环境，77不安装客户端。

### 5.0.1 2026-11-12现场验证进展

本轮证据保存在69的/home/d00937186/bench/results/，两台目录非共享；D服务日志位于77同名目录。以下不是80/20性能验收结果。

- AISBench源码固定d99493ac1914ff41415196eece58f8f296f9e2d6，前缀工具固定ac06bd619726663ac8c7b0c84c9b3cdbb041b062。69现有容器内独立env安装；Torch固定2.6.0，运行时清除PYTHONPATH，pip check通过，两个CLI及GLM tokenizer可用。
- 两端测试副本使用--no-enable-prefix-caching、max-num-seqs=16、P1/D5，保留原启动文件。l1-off-20261112中两条3330/32请求成功；本地命中增量0，重复请求外部命中增加3328 token，Meta batch_get成功增加8，首次DRAM分配增加182214656字节。
- calibration-128k-20261112：131072输入/1024输出成功，两端抢占增量0；新增1024个对象，DRAM增加7176454144字节。SSD allocated不是唯一KV大小或物理读取证据。
- calibration-prefixes-20261112：4个独立117964-token共享前缀，每个增加921个完整块和6454603776字节DRAM；1/2/4个累计占用分别为6454603776、12909207552、25818415104字节，未发生淘汰。一个完整块7008256字节；仅适用于本轮配置。
- concurrency2-20261112：两条131072/1024请求并发成功，两端抢占增量0。暂选并发2；未验证并发4以上。P端启动记录的最大长度KV并发约3.74，服务max-num-seqs=16不等于128K可用并发16。
- 指标目录沿用旧/dev/shm/vllm_metrics，存在旧cache_config_info标签；本轮使用前后差值并结合实际进程参数，不将旧标签认作当前配置。正式对比需隔离指标目录。
- Meta可见DRAM/SSD命中计数和rewarm_bytes；UBS日志显示read cache关闭，但write cache存在。诊断库包含sdk cachehit及内存/磁盘比例接口，尚未找到可用诊断CLI及字节口径说明；不能以Meta SSD标签、allocated或设备读字节单独证明20%物理SSD KV读取。
- AISBench实际负载已验收：results/aisbench/20261112_010429，成功2、失败0，总输入262144、总输出2048。chat模板增加12 token，使用独立校准数据aisbench-128k-chat-calibrated.jsonl（SHA-256 e2df240b32fc4e64fe0f553f319e36d1d5129c942449df4d55687749e4b1d790），原生成数据不覆盖。本轮未控制介质布局，不用于80/20或性能收益结论。
- 介质诊断更新：16个独立长前缀串行预热全部成功，触发3928次淘汰。cold-read-20261112中921个对象回温成功、失败0，Meta回温6451774464字节，与两台专用分区新增扇区数乘512之和完全一致；窗口分区写计数不变。此证据证明本次回温实际发生SSD读取，不再仅依赖诊断CLI。
- mix80-diagnostic-20261112：4热+1冷、串行5请求，每条实际131072/1024；P端每条外部命中117888 token、本地命中0。前4条无回温，最后一条回温6451774464字节。按921块、每块实际KV载荷7005184字节换算，外部读取32258872320字节，SSD占20%。两台分区新增物理读字节合计同样6451774464，写计数不变。载荷大小由同布局921对象回温标定，与DRAM分配每块7008256字节区分；该窗口有介质证据，但只是串行诊断，未完成并发2三轮性能验收。
- 当前Meta累计mount40、unmount24，16个现存rank与本轮P/D存活PID逐一对应，未发现旧LocalService残留；核查文件为两台bench/results/registration-audit-20261112。
- 待完成：并发2下重建并验证各轮80/20布局、池化和无池化基线各三轮有效对比。冷前缀回温后不可直接重复作为冷请求；热请求也需保持独有后缀，避免变成整请求复用。未给出性能收益结论。

### 5.1 固定测试口径

| 项目 | 约定 |
| --- | --- |
| 输入 / 输出 | 131072 / 1024 token；检查实际请求模板和服务返回usage |
| 90%重复率 | 每条输入约90%为共享前缀，不是90%的请求完全重复 |
| 前缀数量 | 独立前缀种类数N；容量标定后确定，不预先拍定 |
| 80/20 | 测量窗口内，外部池读取KV字节中DRAM约80%、物理SSD约20% |
| 允许偏差 | SSD占比20%±2个百分点 |
| 固定项 | 模型、量化、TP8+TP8、P MTP1/D MTP5、最大长度、并发上限、请求顺序、seed、工具版本 |
| 并发 | 当前正式比较暂定2；1、2已通过，4以上须额外容量与安全验证 |

128K+1K=132096，小于当前max-model-len=135000，但仍需计入chat模板、特殊token和投机运行要求。生成器标称长度不等于线上实际长度。输出固定1024时核验ignore_eos及实际输出，思考token也计入输出预算。

当前测试副本已统一max-num-seqs=16；原运行脚本保留20供追溯。正式比较使用测试副本并核对实际进程参数。不得把不同配置的结果直接比较。

### 5.2 【测试环境】准备AISBench与数据工具

本现场使用69现有vllm-pool容器内/home/d00937186/bench/env独立Python环境；不重建容器，不向推理服务的全局Python安装依赖。仅需一套压测客户端，77无需重复安装。

以下为人工准备命令，执行前选择固定commit；目录已存在时先检查，不覆盖已有源码或虚拟环境：

```bash
mkdir -p /home/d00937186/bench
cd /home/d00937186/bench
python3 -m venv env
source env/bin/activate
git clone https://github.com/AISBench/benchmark.git
git clone https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git
git -C benchmark checkout <AISBench_commit>
git -C aisbench_auto_tools_prefix checkout <工具_commit>
python -m pip install -e ./benchmark --use-pep517
python -m pip install -r benchmark/requirements/api.txt -r benchmark/requirements/extra.txt
```

依赖以选定版本的[AISBench安装说明](https://github.com/AISBench/benchmark#-工具安装)为准；核对数据工具的imports，缺项只安装到上述独立环境。tokenizer须支持本机GLM权重，不能为此升级推理容器的transformers。GSM8K源数据按工具要求准备。

```bash
command -v ais_bench
python -m pip show ais_bench_benchmark
python -c 'import ais_bench.benchmark; print(ais_bench.benchmark.__file__)'
ais_bench --help
git -C /home/d00937186/bench/benchmark rev-parse HEAD
git -C /home/d00937186/bench/aisbench_auto_tools_prefix rev-parse HEAD
python -m pip freeze > /home/d00937186/bench/environment.txt
```

本轮安装使用Torch 2.6.0，避免ARM客户端默认引入不需要的CUDA依赖；具体依赖清单保存于environment.txt。源码通过固定提交归档传入，没有.git时核对source-versions.txt及归档哈希，不以rev-parse失败判为缺失。

检查CLI与模块属于同一环境。工具需要--num-warmups 0及对应API配置类；不兼容时先选定兼容版本，不能静默移除预热控制。

修改数据工具config.py：

```python
DATASET_PATH = "/home/d00937186/bench/datasets"
WORK_PATH = "/home/d00937186/bench/benchmark"
MODEL_NAME = "glm"
MODEL_PATH = "/home/weights/GLM-5.3-w4a4c8-mxfp4"
HOST_IP = "192.168.45.69"
HOST_PORT = "9000"
POD_INFO = ["192.168.45.69:8100"]
OUTPUT_DIR = "/home/d00937186/bench/results"
```

工具会修改WORK_PATH内的配置与数据软链接，必须使用独立AISBench源码目录；不同测试不可同时使用这一目录。PD场景dp=1，POD_INFO指向P而非proxy。

### 5.3 【两台容器内】准备基线与池化脚本

在工作目录中复制两套测试脚本，保留已通过短请求验证的原脚本。两组均把--enable-prefix-caching替换为--no-enable-prefix-caching，其他参数一致。

| 配置 | 基线组 | 池化组 |
| --- | --- | --- |
| MooncakeConnectorV1 | 保留P→D传输 | 保留P→D传输 |
| AscendStoreConnector | 从MultiConnector的connectors列表移除 | 保留MemCache后端 |
| 本地prefix caching | 关闭 | 关闭 |
| MTP | P1 / D5 | P1 / D5 |
| 预热 | 完成相同编译与运行时预热，不计入结果 | 另做外部池数据预热，不计入结果 |

基线不能只关闭SSD，否则仍可能命中DRAM池。保留原UB拓扑包装，但基线启动入口不应触发SSD初始化；为测试副本明确选择入口和配置，不直接复用默认执行原脚本的start-pool.sh。此步骤需要生成并审阅测试脚本后再运行。

切换时先等待请求结束，停止proxy及本任务P/D；需要修改Meta配置时再停Meta。仅对确认归属的PID发TERM，确认端口/NPU释放后按Meta→P/D→proxy启动。force_new_disk=true会清空指定缓存分区，每轮池化测试须重新准备缓存，不能假定重启后仍保留。

### 5.4 验证关闭L1后仍能读写池

1. 池化组使用--no-enable-prefix-caching启动，先发短请求确认PD正常。
2. 预热一个可覆盖多个完整缓存块的前缀，等异步写入完成，再重放相同前缀。
3. 对比P端本地与外部命中计数差值、MemCache写入/读取指标和请求日志。本地复用应为0，外部读取应为正，响应成功。
4. 再进行一条128K/1K请求，核对实际长度、NPU峰值、错误及调度抢占。失败时先定位，不进入正式压测。

```bash
curl -fsS http://192.168.45.69:8100/metrics | grep -E 'prefix_cache|external_prefix'
curl -fsS http://192.168.45.69:8000/metrics | grep -E '^memcache_'
```

用测试前后差值，不用进程生命周期累计比例；查询量为0时记为N/A，结合启动配置判断，不能把0/0当作已证明关闭。不要使用请求级skip_reading_prefix_cache，它在当前版本还会跳过外部池匹配。不要用每请求reset替代正式吞吐测试；reset只用于无在途请求时的功能排查，并保持reset_connector=false。

### 5.5 容量标定与前缀数量

HBM容量限制在途并发；DRAM/SSD容量限制可复用前缀工作集，二者分开计算。

```text
共享前缀长度 Lp ≈ 131072 × 0.9（以分词和完整缓存块为准）
单前缀池占用 S ≈ 预热后池分配量增量 / 新增独立前缀数
共享工作集 W ≈ N × S
HBM峰值 ≈ 权重 + 在途请求KV + 激活/图/MTP/通信开销
```

先预热1/2/4个独立前缀，输出尽量短，逐步取指标快照；用相同前缀长度拟合S。若已有淘汰、前缀间有共享、存在双份介质副本或后台回收，简单差分不再可靠，须核对块数/对象数或专用统计。不要把DRAM和SSD的allocated简单相加当唯一KV大小。

GLM的MLA、Indexer、MTP及TP副本会影响实际字节数，不直接套普通Attention公式或盲乘TP8。记录实际块大小、分片/复制关系，并给独有后缀、对齐及元数据留余量。

```bash
curl -fsS http://192.168.45.69:8000/metrics \
  | grep -E '^memcache_(total_capacity_bytes|allocated_bytes|segment_allocated_bytes)'
npu-smi info
```

### 5.6 构造80% DRAM / 20% SSD读取

evict_threshold_high/low是L2占用的高低水位，不是读取比例。建议候选70/60，需写入l3/mmc-meta.conf并在受控重启后确认生效；当前现场此前未显式设置，不能视为已应用。淘汰粒度还需按当前版本指标核对，不用全局128 GiB换算替代实际判断。

1. 按容量标定结果划分热前缀H、冷前缀C和填充数据F，给每个前缀分配稳定ID。
2. 先预热C，再预热F/H，触发C从L2淘汰；确认SSD写入完成。
3. 检查H仍驻留DRAM、C确实需要SSD读取，且C未只停留在UBS IO内存缓冲层。
4. 生成固定测量请求序列；等长等访问次数时可从4:1热冷请求比例起步，最终以实际读取字节比为准。
5. SSD数据回温会改变分布。使用足够大的冷前缀轮换集合，或划分短测量窗口并在窗口外重新准备缓存；记录窗口边界。
6. 每窗口计算介质读取差值；超出预定容差的窗口不计作80/20结果。若没有可信的介质读取指标，先补采集能力，不宣称比例达成。

```text
SSD读取占比 = 本窗口SSD读取KV字节 / 本窗口外部池读取KV总字节
DRAM读取占比 = 本窗口DRAM读取KV字节 / 本窗口外部池读取KV总字节
```

统一按应用KV读取口径，不能将SSD物理I/O字节与DRAM逻辑KV字节直接相加。若UBS IO缓冲独立计数，应单列或明确计入DRAM，并在报告中声明。系统iostat只能辅助证明设备I/O，不能单独证明目标请求的20% KV来自SSD。

### 5.7 【测试环境】数据与压测命令

数据生成与压测使用[前缀工具](https://github.com/rayn-zzz/aisbench_auto_tools_prefix)。正式数据生成一次并保存文件哈希、seed、前缀分组及请求顺序，基线和池化组复用同一文件。completions与chat接口及模板必须一致；本轮chat增加12 token，校准后的chat数据不能直接作为completions的131072-token数据。该工具没有DRAM/SSD定向放置功能，80/20准备按§5.6独立完成。

生成/小规模校验的命令模板（会发起推理，替换占位符后执行）：

```bash
unset PYTHONPATH
source /home/d00937186/bench/env/bin/activate
cd /home/d00937186/bench/aisbench_auto_tools_prefix
nohup python3 aisbench_test.py \
  --input_len 131072 --output_len 1024 \
  --data_num <请求条数> --concurrency 1 --request_rate 0 \
  --dataset_type prefix_cache --repeat_rate 90% \
  --prefix_num <独立前缀数> --seed 1 --dp 1 --npu_num 16 \
  > /home/d00937186/bench/data-check.log 2>&1 &
```

```bash
tail -f /home/d00937186/bench/data-check.log
```

不要直接加--prefix_test：工具在dp=1时将预热并发设为prefix_num；128K多前缀可能超出运行容量。应把预热独立为受控小并发步骤，也避免全量自动预热打乱热冷布局。

正式压测复用已生成的GSM8K格式文件，执行前设置config.py中的OUTPUT_DIR为本轮独立目录：

```bash
nohup python3 aisbench_test.py \
  --dataset /home/d00937186/bench/datasets/<固定数据文件>.jsonl \
  --output_len 1024 --concurrency <并发数> --request_rate 0 \
  --test_type stream --npu_num 16 \
  > /home/d00937186/bench/<组别-并发-轮次>.log 2>&1 &
```

```bash
tail -f /home/d00937186/bench/<组别-并发-轮次>.log
```

此路径不自动采集前缀命中摘要，须独立保存P/D及Meta指标。工具的npu_num=16是双机总卡数，所得单卡吞吐不是P端8卡吞吐。固定数据路径、提示模板及服务usage均需复核；检查AISBench原始失败数，不能只看外层进程退出码。

### 5.8 采集、对比与验收

每轮在工作目录建立独立结果目录，保存开始/结束时间及指标快照；下例在每轮开始前、结束后分别执行，并修改文件名前缀：

```bash
mkdir -p /home/d00937186/bench/results/<轮次>
curl -fsS --max-time 15 http://192.168.45.69:8100/metrics \
  > /home/d00937186/bench/results/<轮次>/before-p.prom
curl -fsS --max-time 15 http://192.168.45.77:8200/metrics \
  > /home/d00937186/bench/results/<轮次>/before-d.prom
curl -fsS --max-time 15 http://192.168.45.69:8000/metrics \
  > /home/d00937186/bench/results/<轮次>/before-meta.prom
```

检查HTTP状态及文件完整性，采集失败记为缺失，不填0。长测量窗口需增加周期快照以检查80/20漂移；分介质读取指标名称与口径须在§5.6确认后加入采集。

| 项目 | 必须记录 |
| --- | --- |
| 服务 | 实际启动参数、镜像/代码版本、MTP、L1开关、池配置与容量 |
| 数据 | 数据哈希、实际输入/输出长度、前缀数、共享比例、请求顺序 |
| 性能 | 成功率、请求/s、输入/输出token/s、TTFT及TPOT分位数 |
| P端收益 | 相同负载窗口的逻辑输入token/s；注明包含池化复用token，区别于实际计算token/s |
| 缓存 | 本地/外部命中差值、DRAM/SSD读取字节比、写入及淘汰证据 |
| 资源 | 两端NPU/HBM峰值、主机内存、SSD I/O、队列和抢占 |

收益=(池化组吞吐/基线吞吐−1)×100%。至少3轮有效结果，报告中位数和波动；预热、缓存布局准备不计入测量时间。两组使用相同并发/到达率；若比较SLO约束下最大吞吐，先固定TTFT/TPOT目标再分别找上限。

AISBench经proxy测得的是端到端吞吐，不能直接命名为纯P吞吐。必须结合P侧指标与D侧队列判断瓶颈；若D饱和，应单列P阶段改善，不能据端到端结果否定或夸大P收益。

验收要求：请求无失败、实际长度符合要求、L1无复用、池化读写有效、80/20有介质证据且在容差内、无OOM。任一项不满足时仅报告功能/诊断结果，不作为正式性能结论。

### 5.9 收尾

停止压测客户端，等待在途请求结束，保存结果、配置与服务日志。需要释放时按proxy→P/D→Meta顺序仅停止本任务进程，并检查NPU及端口；保留容器、权重、分区和测试证据，不清盘。需要继续人工测试时保留服务并明确报告。


## Knowledge Extraction

- [ ] 已记录128K/1K与单窗口分介质诊断；待完成并发2的三轮基线对比后复核可复用结论，不提前提炼性能收益。
