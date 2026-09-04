# Transfer Repair v2 数据门结果：UKCOVID 不支持严格跨 Cohort 训练

日期：2026-09-04  
协议：`docs/TRANSFER_REPAIR_V2_PREREG_ZH.md`  
结果：**NO_GO；没有读取 embedding／projector／matched score，没有启动 Repair v2 训练。**

## 一句话结论

> UKCOVID Standard train 的 recruitment source 与 COVID label 几乎确定绑定，使“同疾病、跨
> source positive”和“异疾病、同 source 且 nuisance-matched negative”无法同时覆盖训练人群。
> 20,714 个 anchor 中只有 51 个（0.246%）同时拥有两类合法 candidate，远低于事前冻结的
> 80%，因此不能在 UKCOVID 上把该目标称为 cross-cohort disease contrastive repair。

## 1. Source × label 结构

| recruitment source | COVID− | COVID+ |
|---|---:|---:|
| REACT | 13,244 | 51 |
| Test and Trace | 0 | 7,419 |

Test and Trace 没有 COVID−。因此两个 source 都含两个 label 的必要条件失败。

## 2. Candidate coverage

冻结定义要求每个 eligible anchor 同时具备：

- 至少 2 个同病、跨 source positive；
- 至少 1 个异病、同 source／sex／age／cough／no-symptom negative。

| population | N | cross-source positive OK | strict negative OK | 两者同时 OK |
|---|---:|---:|---:|---:|
| 全部 train | 20,714 | 7,470（36.06%） | 5,760（27.81%） | **51（0.246%）** |
| COVID− | 13,244 | **0（0%）** | 5,709（43.11%） | **0（0%）** |
| COVID+ | 7,470 | 7,470（100%） | 51（0.683%） | **51（0.683%）** |

这不是“样本略少”，而是正负 candidate 的可用性落在互不重叠的人群上：

- REACT 阴性有同 source 阳性 negative candidate，但没有跨 source 的同病 positive；
- Test-and-Trace 阳性有跨 source 同病 positive，但没有同 source 阴性 negative；
- 只有 51 个 REACT 阳性同时满足两边。

## 3. 为什么不能使用 fallback 强行训练

若放松为任意同标签、不同 participant，100% anchor 都有 positive；若只要求同 source 的异标签，
64.18% anchor 有 negative。但这两个数字不参与 gate：

- 允许 same-source positive 后，模型仍可用 cohort shortcut，核心“跨 cohort”约束消失；
- 允许随机异标签 negative 后，模型可继续利用 source／年龄／症状区分正负；
- 将 matched test participant 加入 candidate pool 会造成训练--测试污染。

所以不能把 fallback 版本继续叫 Repair v2，也不能为得到一个训练结果而修改 80% 门槛。

## 4. 这个结果说明什么

这是 **data-feasibility failure**，不是模型负结果：方法尚未被训练，因此不能推断 Repair v2 有效或
无效。它说明 UKCOVID 恰恰因为 source--label coupling 太强，不具备训练 disease-invariant
cross-cohort objective 所需的 common support。

这一点与 audit 主结论一致：训练数据把 cohort 与疾病绑定得过紧；但 audit 可以测量该问题，
不代表同一份 source train 也足以监督模型解除它。

## 5. 下一步边界

UKCOVID Repair v2 在本协议下停止。若继续，应转向在训练集中确实存在多 country／site × 两类疾病
共同支持的数据集，并在任何 embedding 或模型分数前运行同一数据门。CODA TB 是首选候选，
Cambridge Task-2 可作备选；只有数据门通过，才实现 sampler、`L_meta + L_cross` 和 AST 五 seed
训练。

公开的 aggregate-only 结果：`results/transfer_repair_v2_pair_gate.json`，SHA-256：
`8cae8c54d4c8a61fd1fe4c56d324bc17d5802fc66ef86949741e7068ce16ea54`。
