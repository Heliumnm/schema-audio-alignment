# Transfer Repair v2：保留 Raw 的跨 Cohort 疾病对比学习

日期：2026-09-04

状态：**训练前冻结；第一步仅运行 model-blind pair-feasibility gate。**

本协议根据既有 audit 与失败的 Repair v1 提出。Repair v1 只能降低部分录音伪影，几乎不能降低
性别、年龄或招募来源的可解码性，也没有改善 matched COVID transfer。因此 v2 不再事后删除
若干线性方向，而是在训练时改变 disease contrast 的正负样本。

由于既有 matched 结果已经参与本方法设计，v2 是探索性 mitigation，不是 untouched confirmation。

## 1. 科学问题

> 在保留 same-participant audio--metadata alignment 的同时，加入“同疾病、跨 cohort”的音频
> 正样本和“异疾病、同 nuisance stratum”的音频负样本，能否减少患者／cohort correspondence，
> 并改善 covariate-balanced disease transfer？

## 2. 冻结表示与模型

首轮只使用 UKCOVID + AST-6L。音频编码器和 Phi-2 文本编码器全部冻结。

```text
audio -> frozen AST-6L -> raw representation r
                              |
                              v
                      trainable projector g
                              |
                              v
                      repaired representation z

downstream representation = concat(r, z)
```

Projector 复用原 Correct arm 的结构、初始化方式和训练预算；v2 从头训练，不从 Correct checkpoint
继续训练。Raw preservation 是下游拼接，不是向量相加；因此 raw 音频信息始终可直接到达 readout。

## 3. 两项训练目标

### 3.1 Correct metadata loss

保留原始单向 audio-to-text contrastive loss：同一 participant 的 audio 与 metadata text 为正配对，
batch 内其他文本为分母。记为 `L_meta`。

### 3.2 Cross-cohort disease loss

对 Standard-train anchor `i`，定义：

```text
P(i) = {j: y_j = y_i, source_j != source_i, participant_j != participant_i}

N(i) = {k: y_k != y_i,
           source_k = source_i,
           sex_k = sex_i,
           age_bin_k = age_bin_i,
           cough_any_k = cough_any_i,
           symptom_none_k = symptom_none_i}
```

缺失值是显式类别 `[MISSING]`，不能在匹配时通配。每个 eligible anchor 每 epoch 固定抽取：

- 2 个不同的 `P(i)`；
- 8 个 `N(i)`；候选不足 8 时允许有放回抽样，但至少必须存在 1 个候选。

跨 source positive 内部按 `(sex 是否不同, age_bin 是否不同)` 的差异数优先：先从差异数最高且至少
有两个候选的层级抽样；若没有任何层级含两个候选，则该 anchor 不计算 `L_cross`。这不会放松
“跨 source”条件。

`L_cross` 对每个 anchor 使用两个 positive 的 log-sum-exp 为分子，两个 positive 与八个 matched
negative 为分母。相似度是 L2-normalised `z` 的 cosine similarity，温度沿用 `0.07`。

总目标固定为：

```text
L_total = L_meta + 1.0 * L_cross
```

不搜索 lambda、positive 数量、negative 数量、温度或匹配字段。

## 4. Model-blind pair-feasibility gate

训练前只能读取 Standard-train 的 participant ID、COVID label、recruitment source、sex、age、
`symptom_cough_any` 和 `symptom_none`。不得读取任何音频 embedding、projector、retrieval、疾病
readout、matched 或 matched-long 模型分数。

必须同时满足：

1. 全部 Standard-train anchors 中至少 80% 同时拥有至少 2 个合法 `P(i)` 和至少 1 个合法
   `N(i)`；
2. COVID+ 和 COVID- 两类各自的上述覆盖率均至少 80%；
3. 两个 recruitment source 中都必须有两个 disease label；
4. 所有 candidate 都来自 Standard train，participant 自配对数为 0；
5. 输出只保存 aggregate counts，不保存 participant ID。

任何一项失败即为 `NO_GO`：不允许通过退回 same-source positive、随机 opposite-label negative、
移除 source matching 或纳入 matched participants 来启动 UKCOVID v2。

## 5. 数据门通过后的 deterministic sampler

- batch size 64，`drop_last=False`；
- 每个 epoch 覆盖全部 Standard-train participant 一次；
- `L_meta` 对全部 anchor 计算；
- `L_cross` 只对 gate 定义的 eligible anchor 计算；
- positive／negative sampling 每 epoch 重采样，但由 `seed + epoch` 完全确定；
- 五个 seed：0--4；500 epochs；Adam `lr=1e-3`，无 scheduler、无 weight decay；
- 只使用 epoch 500，不按 source 或 target 结果挑 checkpoint。

## 6. 冻结比较

| arm | 作用 |
|---|---|
| Raw | 冻结音频基线 |
| Correct | 原 same-participant metadata alignment |
| Within-label | label-level pairing control |
| Raw + Correct | 保留 raw 的原 alignment |
| Raw + Repair v2 | 主方法 |

主比较为 `Raw + Repair v2 - Raw + Correct`；关键次要比较为 `Raw + Repair v2 - Raw`。

## 7. 分阶段运行

1. **Gate**：只跑上述 pair-feasibility；
2. **S0**：合成数据验证 sampler、loss 手算一致、无跨 split candidate；
3. **S1**：一个 seed、少量 update，只检查 loss 下降、有限值、有效秩和配对 hash；
4. **Formal AST**：五 seed × 500 epochs；训练与 source validation 冻结后，才一次读取 matched；
5. 只有 AST 主 gate 通过，才运行 OPERA-CT／HeAR；
6. 至少两个 backbone 支持后，才在 CODA TB 或 Cambridge 中选择一个外部数据集。

## 8. 事前成功标准

AST 主 gate 同时要求：

1. matched `Raw+Repair - Raw+Correct` 的 `Delta AUROC >= +0.01`，95% paired CI 下界大于 0；
2. matched-long Delta AUROC 同方向；
3. `Raw+Repair` 不劣于 Raw；
4. sex probe 相对 Correct 至少下降 0.03；age、source 全部报告但不逐项设门；
5. source AUROC 相对 `Raw+Correct` 下降不超过 0.02；
6. matched NLL 与 Brier 不显著变差；
7. 表示不坍缩；profile MRR 必须报告。

理想但非硬性条件是 `Correct MRR > Repair MRR > Within MRR`。若 Repair retrieval 接近 Global，
只能说明 alignment 被破坏。

## 9. 成功后的唯一 ablation

只有主 gate 通过才运行 Original／Positive-only／Negative-only／Full Repair 四臂，区分收益来自
跨 cohort positives 还是 matched negatives。主 gate 失败时不运行 ablation，也不根据 matched
结果改 lambda、采样数量或匹配规则。

## 10. 可能的负结果解释

- nuisance probe 下降但 matched 不升：减少 demographic correspondence 不能创造本来很弱的
  acoustic disease signal；
- nuisance probe不降且 matched 不升：该目标没有实质改变表示；
- pair-feasibility gate 失败：当前 source-domain joint distribution 不支持这个科学问题，不能把
  “没有合法训练对”误写成“方法失败”。
