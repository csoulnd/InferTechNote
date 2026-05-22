# 310P 模型与特性支持矩阵

本文档用于统计 **Ascend 310P（Atlas 300I）** 的模型与特性支持现状、规划，并对照现有 E2E 用例给出**可落地的用例更新建议**。

> 平台能力矩阵中，每一格仅表示**该模型是否支持对应特性**，不表示多项特性可同时叠加使用。E2E **不要求**对矩阵穷举覆盖，以少量代表用例兼顾效率；**4 卡 Light Job 的 4 个用例**是当前推荐采样范式（稠密 / MoE / 线性注意力 / VLM + TP + 量化）。

### W8A8 量化说明

| 模型族 | W8A8 实际类型 |
|---|---|
| Qwen3 稠密、Qwen3-VL 稠密 | **W8A8SC** |
| 其余模型（Qwen3 MoE、Qwen3-VL MoE、Qwen3.5 / Qwen3.6 全系等） | **W8A8 Dynamic** |

### 图例（平台能力矩阵）

| 标记 | 含义 |
|:--:|---|
| ✅ | 平台已支持 |
| ❌ | 平台不支持 |
| 🟡 | 平台能力规划中 |
| — | 不适用 |

### 图例（E2E 覆盖矩阵）

| 标记 | 含义 |
|:--:|---|
| ✅ | 已有 E2E 代表用例（含 Light CI） |
| 🟡 | 建议新增或调整用例 |
| ❌ | 平台不支持 / 无需 E2E |
| — | 当前不要求单独 E2E |

### CI 资源池（现有 workflow）

| Job | Runner | 卡数 | 说明 |
|---|---|---|---|
| `e2e_310p` | `linux-aarch64-310p-1` | 1 | 单卡资源池 |
| `e2e_310p-4cards` | `linux-aarch64-310p-4` | 4 | 多卡资源池 |
| `linux-aarch64-310p-2` | — | 2 | 已注册，**当前无对应 Job** |
| E2E-Full | — | — | 现 `contains_310: false`，**不跑** 310P |

单卡与多卡用例须分别落入上述资源池，**不宜**新增依赖 2 卡 runner 的常规 Light 用例。

---

## 平台能力矩阵

| 特性 \ 模型 | Qwen3<br>Dense | Qwen3<br>MoE | Qwen3-VL<br>Dense | Qwen3-VL<br>MoE | Qwen3.5<br>Dense | Qwen3.5<br>MoE | Qwen3.6<br>Dense | Qwen3.6<br>MoE | Qwen-ASR |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 模型 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 |
| TP | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 |
| ACL Graph | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 |
| W8A8 Dynamic | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| W8A8SC | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | — |
| MTP | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | ❌ |
| Prefix Cache | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
| Function Call | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 |

---

## E2E 覆盖矩阵

当前 Light CI **8** 用例（均 `enforce_eager=True`）。

| 特性 \ 模型 | Qwen3<br>Dense | Qwen3<br>MoE | Qwen3-VL<br>Dense | Qwen3-VL<br>MoE | Qwen3.5<br>Dense | Qwen3.5<br>MoE | Qwen3.6<br>Dense | Qwen3.6<br>MoE | Qwen-ASR |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 模型 | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — | — |
| TP | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — | — |
| ACL Graph | 🟡 | — | — | — | — | — | — | — | — |
| W8A8 Dynamic | ❌ | 🟡 | ❌ | — | — | — | — | — | — |
| W8A8SC | ✅ | ❌ | — | ❌ | ❌ | ❌ | ❌ | ❌ | — |
| MTP | ❌ | ❌ | ❌ | ❌ | — | — | — | — | ❌ |
| Prefix Cache | — | — | — | — | — | — | — | — | — |
| Function Call | — | — | — | — | — | — | — | — | — |

---

## 用例更新建议

以下仅为文档建议，**本文档不修改 workflow**；实施时需分别改 `tests/e2e/310p/` 与 `_e2e_test.yaml` / `pr_test_full.yaml` 等。

### 保留（当前 Light CI，8 用例）

**单卡 Job**（`linux-aarch64-310p-1`）

| 用例 | 模型 | 看护要点 |
|---|---|---|
| `test_qwen3_dense_tp1_fp16` | Qwen3-8B | 稠密 + TP1 + FP16 |
| `test_qwen3_dense_tp1_w8a8` | Qwen3-8B-W8A8 | 稠密 + TP1 + W8A8SC |
| `test_qwen3_5_dense_tp1_fp16` | Qwen3.5-4B | 线性注意力稠密 + TP1 |
| `test_qwen3_vl_8b_tp1_fp16` | Qwen3-VL-8B | VLM 稠密 + TP1 |

**4 卡 Job**（`linux-aarch64-310p-4`）— 推荐采样核心

| 用例 | 模型 | 看护要点 |
|---|---|---|
| `test_qwen3_dense_tp4_w8a8` | Qwen3-32B-W8A8 | 大稠密 + TP4 + W8A8SC |
| `test_qwen3_moe_tp4_fp16` | Qwen3-30B-A3B | MoE + TP4 + FP16 |
| `test_qwen3_5_moe_tp4_fp16` | Qwen3.5-35B-A3B | MoE + 线性注意力 + TP4 |
| `test_qwen3_vl_8b_tp2_fp16` | Qwen3-VL-8B | VLM + TP2 |

### 建议删除（2 用例）

| 用例 | 文件 | 建议 | 理由 |
|---|---|---|---|
| `test_qwen3_dense_tp2_fp16` | `multicard/test_dense_model_multicard.py` | **删除** | TP2 无对应资源池 Job；稠密 TP 已由单卡 TP1 + 4 卡 TP4 代表 |
| `test_qwen3_moe_tp2_w8a8` | `multicard/test_moe_model_multicard.py` | **删除** | 同上；Dynamic 量化宜在 **4 卡 Job** 新增 TP4 代表用例，而非保留 2 卡 orphan |

删除后若仍需本地调试 TP2，可通过 `/e2e` 临时加测，不必常驻仓库。

### 建议新增

| 建议用例 | 目标 Job / 资源池 | 优先级 | 说明 |
|---|---|:--:|---|
| 在现有稠密代表上增加 **ACL Graph** 路径（如 `test_qwen3_dense_tp1_graph`） | E2E-Light · 单卡 | P1 | 平台已支持 Graph，Light 仅需 **1 条** 代表；单卡池成本更低 |
| **Qwen3 MoE + W8A8 Dynamic + TP4**（如 `test_qwen3_moe_tp4_w8a8`） | E2E-Light · 4 卡 | P1 | 补 Dynamic 量化代表；与现有 MoE FP16 用例并列，不增加模型族条数 |
| **Qwen3.6** 代表（稠密或 MoE 择一，TP1 或 TP4 择一） | E2E-Light · 单卡或 4 卡 | P2 | 新模型族上线稳定后，**替换**现有某条代表即可，不建议扩容 |
| **Function Call** 代表（Qwen3.5 / Qwen3.6 择一） | E2E-Full · 310P | P2 | 平台已支持，场景较重，适合 Full 而非常规 Light |
| **MTP / Prefix Cache** 代表 | E2E-Full · 310P | P3 | 待平台 🟡 交付后各 **1 条** |
| **Qwen-ASR** 代表 | E2E-Full · 310P | P3 | 待模型交付后新增 |

**暂不建议新增**：Qwen3-VL MoE、各模型族 VL W8A8SC、Qwen3.5/3.6 全量 Dynamic 等矩阵空格——4 卡采样已覆盖主路径，强行补全性价比低。

### E2E-Light vs E2E-Full 分工建议

| 套件 | 310P 现状 | 建议 |
|---|---|---|
| **E2E-Light** | `_310_tracker` 触发；单卡 4 + 4 卡 4 | **维持 8 用例量级**；仅 P1 增 Graph（单卡 1 条）+ MoE Dynamic TP4（4 卡 1 条）；删 2 卡 orphan |
| **E2E-Full** | `contains_310: false` | 合并前深度验证：将 `contains_310` 置 true（或等价配置），在 Full 中跑 **扩展集**——例如在 Light 8 用例基础上增加 Graph 全量、Function Call、以及后续 MTP / Prefix Cache / Qwen-ASR 代表用例 |
| **`/e2e` 评论** | 路由 `310p` → 对应 runner | 保留，用于 PR 调试未入库或实验性用例 |

### 变更后 Light CI 用例目标（参考）

| 资源池 | 现况 | 目标 |
|---|---|---|
| 单卡（4→**5**） | 4 用例 | 保留现有 4 条 + **新增 1 条 ACL Graph 代表**（可选） |
| 4 卡（4→**4~5**） | 4 用例 | 删除 2 卡 orphan 后，**新增 1 条 MoE W8A8 Dynamic TP4**；总量仍控制在 4~5 条 |

---

## 参考

- [E2E CI Test 文档](e2e_ci_test.md)
- [Atlas 300I 使用教程](../../tutorials/hardwares/310p.md)
- 测试目录：`tests/e2e/310p/`
- Workflow：`.github/workflows/pr_test_light.yaml`、`.github/workflows/pr_test_full.yaml`、`.github/workflows/_e2e_test.yaml`
