# 310P 模型与特性支持矩阵

本文档统计 **Ascend 310P（Atlas 300I）** 的平台能力现状与 **E2E-Light CI 覆盖现状**。

> 平台能力矩阵中，每一格仅表示**该模型是否支持对应特性**，不表示多项特性可同时叠加使用。

### 量化说明

310P 上 **W8A8 Static**、**W8A8 Dynamic**、**W8A8SC** 为三种不同量化方案：

| 量化类型 | 适用模型族 | 说明 |
|---|---|---|
| **W8A8 Static** | Qwen3 稠密、Qwen3-VL 稠密 | 权重名中的 `*-W8A8` 对应 static 量化（代码注册为 `W8A8`） |
| **W8A8 Dynamic** | Qwen3 MoE、Qwen3-VL MoE、Qwen3.5 / Qwen3.6 全系等 | 动态 W8A8 量化 |
| **W8A8SC** | Qwen3 稠密、Qwen3-VL 稠密 | 稀疏压缩量化，与 W8A8 Static / Dynamic 并列 |

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
| ✅ | 已支持，且 **E2E-Light CI 已覆盖** |
| ❌ | 已支持，**E2E 未覆盖** |
| 🟡 | **将支持**，E2E **尚未覆盖**（Pending） |
| — | 不支持，无需 E2E 覆盖 |

---

## 平台能力矩阵

| 特性 \ 模型 | Qwen3<br>Dense | Qwen3<br>MoE | Qwen3-VL<br>Dense | Qwen3-VL<br>MoE | Qwen3.5<br>Dense | Qwen3.5<br>MoE | Qwen3.6<br>Dense | Qwen3.6<br>MoE | Qwen-ASR |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 模型 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 |
| TP | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 |
| ACL Graph | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 |
| W8A8 Static | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | — |
| W8A8 Dynamic | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| W8A8SC | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | — |
| MTP | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | ❌ |
| Prefix Cache | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
| Function Call | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 |

---

## E2E 覆盖矩阵

对照 `tests/e2e/310p/` 与 E2E-Light CI（`e2e_310p` / `e2e_310p-4cards`）。当前 Light CI 共 **9** 个用例；E2E-Full 现 `contains_310: false`，不执行 310P 用例。

| 特性 \ 模型 | Qwen3<br>Dense | Qwen3<br>MoE | Qwen3-VL<br>Dense | Qwen3-VL<br>MoE | Qwen3.5<br>Dense | Qwen3.5<br>MoE | Qwen3.6<br>Dense | Qwen3.6<br>MoE | Qwen-ASR |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 模型 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | 🟡 |
| TP | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | 🟡 |
| ACL Graph | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 |
| W8A8 Static | ✅ | — | ❌ | — | — | — | — | — | — |
| W8A8 Dynamic | — | ❌ | — | ❌ | ❌ | ❌ | ❌ | ❌ | — |
| W8A8SC | ❌ | — | ❌ | — | — | — | — | — | — |
| MTP | — | — | — | — | 🟡 | 🟡 | 🟡 | 🟡 | — |
| Prefix Cache | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
| Function Call | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 |

---

## CI 资源池与用例现状

| Job | Runner | 卡数 |
|---|---|---|
| `e2e_310p` | `linux-aarch64-310p-1` | 1 |
| `e2e_310p-4cards` | `linux-aarch64-310p-4` | 4 |

单卡与多卡用例分别运行在上述资源池。`linux-aarch64-310p-2` 已注册，当前无对应 Job。

### E2E-Light CI 用例（9）

**单卡 Job**（跑完整 `test_dense_model_singlecard.py` + `test_vl_model_singlecard.py`）

| 用例 | 模型 | 模式 / 量化 |
|---|---|---|
| `test_qwen3_dense_tp1_fp16` | Qwen3-8B | Eager + FP16 |
| `test_qwen3_dense_tp1_fp16_aclgraph` | Qwen3-8B | ACL Graph（`FULL_DECODE_ONLY`） |
| `test_qwen3_dense_tp1_w8a8` | Qwen3-8B-W8A8 | Eager + **W8A8 Static** |
| `test_qwen3_5_dense_tp1_fp16` | Qwen3.5-4B | Eager + FP16（线性注意力路径） |
| `test_qwen3_vl_8b_tp1_fp16` | Qwen3-VL-8B | Eager + FP16 |

**4 卡 Job**（按用例名逐条执行）

| 用例 | 模型 | 模式 / 量化 |
|---|---|---|
| `test_qwen3_dense_tp4_w8a8` | Qwen3-32B-W8A8 | Eager + **W8A8 Static** |
| `test_qwen3_moe_tp4_fp16` | Qwen3-30B-A3B | Eager + FP16 |
| `test_qwen3_5_moe_tp4_fp16` | Qwen3.5-35B-A3B | Eager + FP16（线性注意力路径） |
| `test_qwen3_vl_8b_tp2_fp16` | Qwen3-VL-8B | Eager + FP16 |

### 仓库内未纳入 Light CI 的用例（2）

| 用例 | 模型 | 说明 |
|---|---|---|
| `test_qwen3_dense_tp2_fp16` | Qwen3-8B | TP2，无对应 CI Job |
| `test_qwen3_moe_tp2_w8a8` | Qwen3-30B-A3B-W8A8 | TP2 + **W8A8 Dynamic**，无对应 CI Job |

---

## 参考

- [E2E CI Test 文档](e2e_ci_test.md)
- [Atlas 300I 使用教程](../../tutorials/hardwares/310p.md)
- 测试目录：`tests/e2e/310p/`
- Workflow：`.github/workflows/pr_test_light.yaml`、`.github/workflows/_e2e_test.yaml`
