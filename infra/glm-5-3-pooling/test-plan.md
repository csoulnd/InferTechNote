---
title: "无池化与池化的配对性能测试方法"
type: design
domain: infra
status: active
---

# 无池化与池化的配对性能测试方法

## 目的与适用范围

在相同模型、接口、负载和推理参数下，比较关闭外部池的基线与开启池化后的端到端性能，重点观察TTFT，同时检查TPOT、吞吐、成功率及缓存来源。控制本地HBM前缀复用，区分外部前缀复用收益与DRAM/SSD介质差异。

本文不指定并发档位，不包含工具安装。使用已准备的AISBench独立环境、原生Prefix Cache插件和原生infer/perf命令，不通过poolbench调度发压。部署见[deployment.md](deployment.md)，目录采用[工作区约定](../../.agents/skills/service-test-workspace/SKILL.md)。

## 参数与case设计

C为并发，N为每轮正式请求数，K为独立前缀数，L/O为输入/输出token长度，H为共享前缀比例。每次实验先固定这些参数、请求顺序、seed、模型/量化/并行、MTP、图模式与工具版本。正式前缀长度按实际缓存块对齐并经tokenizer验证。

| Case | 服务与初态 | 目的 / 验收 |
|---|---|---|
| 基础功能 | 两种部署分别短请求、目标长度请求 | PD成功，长度正确，无OOM或加载失败 |
| 基线 | 关闭HBM前缀复用，移除外部池connector，保留PD传输 | 取得同一冻结数据的无外部池结果 |
| 池化首次写入 | 关闭HBM复用，外部池为空或使用全新前缀 | 验证写入；单列冷态，不混入预热后结果 |
| 池化预热读取 | 同前缀、不同后缀，显式完成缓存预热 | 外部复用符合数据设计，HBM无复用 |
| 池化工作集扫描 | 固定C，改变K或访问顺序 | 覆盖不同实测DRAM/SSD来源区间 |
| 配对性能 | 同C/N/L/O/H/K与数据顺序，基线和池化各多轮 | 比较TTFT及吞吐，检查失败、重试与波动 |

N应与C关联并在实验前固定倍数，例如使用N=qC；q是实验参数，不是工具约束。没有“必须被5整除”的通用要求。K不必为C的倍数；不能均分时记录各组条数，不通过复制完整请求补齐数量。尾部排空导致实际平均并发低于C，不应将配置并发误报为全程恒定并发。

DRAM来源目标由实验定义，可扫描一个区间而非追求固定稳态。调K影响工作集与重用距离，但不能保证某个比例。先标定容量及后缀写入压力；HBM限制在途容量，外部DRAM/SSD限制复用工作集。个别rank满载、碎片、异步回温均可能先于全局容量耗尽。

## 工具与数据准备（不含安装）

先用实际独立环境确认CLI支持哪些参数；不升级推理全局依赖。

```bash
PY=/home/d00937186/bench/env/bin/python
CONTAINER=vllm-pool
PLUGIN=/home/d00937186/bench/benchmark/plugins/prefix_cache
docker exec "$CONTAINER" "$PY" -m ais_bench.benchmark.cli.main --help
docker exec -e PYTHONPATH="$PLUGIN" "$CONTAINER" "$PY" -m ais_bench_prefix_cache.cli --help
```

原生Scenario关键字段：

| 字段 | 用途 |
|---|---|
| requests.count | 正式数据总数，多轮可一次生成再分割 |
| requests.input_length / output_length | 固定或受控长度分布 |
| prefix_cache.mode=warmup | 为目标前缀生成独立预热计划 |
| prefix_cache.target_hit_rate | 理论前缀复用比例，不是DRAM来源比例 |
| prefix_cache.groups.count | K个独立前缀 |
| prefix_cache.groups.assignment | 组间请求分配 |
| prefix_cache.order.strategy | 固定顺序，例如interleave轮询 |
| run.random_seed | 数据生成可重复性；不同候选避免完整输入重复 |
| service.dp_size | 前缀处理侧实际DP数量，不能直接填D侧DP或PCP数量 |

```bash
# SCENARIO、MANIFEST为实际文件的绝对路径
docker exec -e PYTHONPATH="$PLUGIN" "$CONTAINER" "$PY" -m ais_bench_prefix_cache.cli inspect --scenario "$SCENARIO"
docker exec -e PYTHONPATH="$PLUGIN" "$CONTAINER" "$PY" -m ais_bench_prefix_cache.cli prepare --scenario "$SCENARIO" --mode text
docker exec -e PYTHONPATH="$PLUGIN" "$CONTAINER" "$PY" -m ais_bench_prefix_cache.cli validate --manifest "$MANIFEST"
```

从prepare输出获得manifest路径；保留原生requests/full/manifest。若拆成多轮，原始产物不改写，拆分配置指向各轮独立文件。不同轮次共享相同前缀，但独有后缀不重复；基线与池化配对使用同一轮数据。原生warmup.plan独立于正式请求，可导出为CustomDataset使用infer发送，不需要自定义发压器。

插件在线run的reset/分析面向vLLM前缀缓存；未验证其适配外部池前，不把它当作外部池清理与介质比例验收入口。

## 基线测试步骤

1. 核验服务归属，等待在途完成，按部署文档切换到baseline。保留PD connector，关闭本地前缀复用，不接AscendStore外部池；只关SSD不算无池化基线。
2. 检查P/D/proxy健康和目标长度请求。使用独立暖机数据覆盖目标输入/输出形态，等待编译完成；暖机不计正式性能。
3. 固定C/N与冻结数据，执行正式perf，每轮使用独立原生结果目录。
4. 检查请求全部成功、实际长度、内部重试、端到端TTFT/TPOT和吞吐。失败轮保留，不用加重试掩盖失败后宣称有效。

## 池化测试步骤

1. 按部署文档启动Meta→P/D→proxy，关闭HBM前缀复用，保留相同模型及推理参数。确认实际注册容量、SSD与UBS配置。
2. 建立并说明初态。服务启动、外部池初始化、使用新seed是不同操作；仅重启vLLM或调用reset_prefix_cache不能假定整个外部池已清空。
3. 使用独立数据做目标形态算子暖机，再对K个目标前缀做缓存预热，确认写入完成及请求成功。插件短输出预热不能代替完整decode形态暖机。
4. 使用与基线对应轮次完全相同的数据、顺序、C/N执行perf。每轮重新按相同协议准备目标前缀；不得重发旧完整输入冒充约定的部分前缀复用。
5. 固定C调整K，先做功能/容量试探，再补足有效性能轮次；超出目标介质区间、前缀缺失或回温失败的窗口单列，不作为该区间的正式结果。

一次启动后多轮代表连续服务窗口，前轮后缀写入和驱逐会改变后轮初态。独立重复实验应逐轮重建初态；连续窗口则逐轮报告实际状态和比例，不宣称初态完全相同。每个有效性能点至少三轮，短样本P99只描述该轮，不作稳定尾延迟保证。

## 原生手动命令

配置中的CustomDataset路径、模型接口、stream、max_out_len及生成参数应事先核验；stream开启才能测TTFT。固定输出长度需检查ignore_eos和实际输出。C、N、K、ROUND由操作者设置，不预设档位。

```bash
W=/home/d00937186/glm53-pooling-lab
# CFG为该轮test-configs目录；MODE为baseline或pooling；CASE、ROUND须唯一
OUT="$W/log/tests/$MODE/$CASE/$ROUND"
test ! -e "$OUT"
mkdir -p "$OUT/warmup" "$OUT/measurement"

# 独立预热：暖机/缓存预热配置和条数按阶段选择
docker exec "$CONTAINER" "$PY" -m ais_bench.benchmark.cli.main "$WARMUP_CONFIG" --mode infer --batch-size "$C" --num-prompts "$WARMUP_N" --num-warmups 0 -w "$OUT/warmup/results" > "$OUT/warmup/console.log" 2>&1
# 人工确认预热全部成功、无回温异常后继续

docker exec "$CONTAINER" "$PY" -m ais_bench.benchmark.cli.main "$MEASUREMENT_CONFIG" --mode perf --batch-size "$C" --num-prompts "$N" --num-warmups 0 -w "$OUT/measurement/results" > "$OUT/measurement/console.log" 2>&1

tail -f "$OUT/measurement/console.log"
```

`--num-warmups`使用正式数据开头的请求，超过数量时循环；其结果不计性能，但会改变外部缓存，故部分前缀复用实验显式设0。无池化或明确接受完整请求复用的实验可以按目的使用原生warmup。CLI支持多档并发并不意味着共享数据连跑就具有相同缓存初态，本方案按case手动执行。

## 日志、介质证据与结果解释

服务日志每次启动只存一份完整原始输出；各轮仅保存AISBench原生日志及结果。用服务日志请求ID和原生时间定位窗口，不默认切片、压缩、周期采样或生成助手调度/元数据。必要对应信息只写README，分析由用户手动触发。

外部前缀命中率=外部复用token/总输入token；DRAM与SSD来源比例的分母仅为外部复用KV量，重新计算部分另列。P日志能给外部命中量，不能单独区分介质；Meta Get计数可能包含重复查询和回温后读取；UBS内存缓冲不是物理SSD读取，SSD→DRAM→HBM不能重复算两次来源。

现有原生日志能否完成逐请求/逐KV互斥来源归因，需先核对字段、时间边界及字节口径。缺少证据时介质比例标为未核验；不能凭TTFT变快、静态请求分配、模拟值或回温计数直接宣布真实比例。若用户授权增加采集，另说明采集范围和口径；不自动加入正式日志流程。

原生结果通常在results/<时间戳>/performances/<模型标签>/，保留CSV/JSON、逐请求明细和原生控制台日志。报告成功率、实际长度、TTFT/TPOT分位数、输出吞吐及各轮波动。proxy测到的是端到端性能，不直接命名为纯P性能；结合PD排队辨别瓶颈。无有效介质证据的窗口可以保留性能观测，但不能称已完成特定介质区间验收。

## 参考与知识沉淀

- [AISBench原生前缀插件](https://ais-bench-benchmark.readthedocs.io/en/latest/advanced_tutorials/prefix_cache.html)
- [AISBench仓库](https://github.com/AISBench/benchmark)

## Knowledge Extraction

- [ ] 原生流程与介质归因充分验收后，再提炼可复用结论。
