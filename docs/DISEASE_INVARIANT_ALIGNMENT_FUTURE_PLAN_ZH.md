# 下一阶段方法：Disease-Invariant Positive-Pair Alignment

日期：2026-08-23

状态：**方法设计；尚未实现、尚未训练、没有结果。**

与当前 ICASSP audit 的关系：作为 Discussion/Future Work，不得混入已完成结果。

## 1. 为什么不能直接“把同标签样本都当正样本”

当前 audit 说明 correct metadata alignment 学到了患者 correspondence，但没有稳定疾病 transfer。
直接把所有 COVID 相同的样本当正样本并不自动解决问题：在 UKCOVID Standard train 中，标签与
招募来源几乎相同，same-label positive 仍然可能只是 same-cohort positive。

因此下一阶段目标不是简单从 instance-level 换成 label-level，而是：

> **同疾病、跨 nuisance strata 的样本才构成主要正样本；不同疾病、nuisance 匹配的样本构成
> 关键负样本。患者和 cohort metadata 只用于构造约束，不再作为音频必须复原的文本目标。**

## 2. 先把 metadata 拆成三类

| 类型 | 例子 | 在训练中的角色 |
|---|---|---|
| `A`：可听声学证据 | wheeze、crackle、频带、持续时间、音质 | 只有可靠标注／teacher gate 通过后，才能作为 audio--text positive target |
| `D`：疾病目标 | COVID／疾病标签 | 定义 disease-level positive set，不伪装成自然语言知识 |
| `N`：患者／cohort nuisance | 年龄、性别、症状、吸烟、来源、设备、时长 | 只用于平衡、跨层采样和 nuisance probe，不与音频硬对齐 |

当前数据边界：COUGHVID teacher feasibility 没通过，UKCOVID 也没有可靠的事件级可听证据文本，
所以首版不能声称实现了 `audio <-> audible report`。它只能测试 disease-invariant representation
learning。真正的 audio--language 版本必须等待可靠的 `A` 类标注或外部数据。

## 3. Positive／negative pair 的冻结定义建议

对 anchor `i`，标签为 `y_i`，nuisance stratum 为
`s_i=(source, sex, age_bin, symptom_bin)`：

### Positive set

```text
P(i) = {j: y_j = y_i, participant_j != participant_i,
           source_j != source_i,
           nuisance_distance(s_i, s_j) >= d_min}
```

也就是相同疾病、不同参与者，并优先跨招募来源／人口学层。若某标签没有足够跨来源样本，
该 anchor 不进入 contrastive loss，不能退回同来源正样本补数量。

### Hard negative set

```text
N(i) = {k: y_k != y_i,
           source_k = source_i,
           sex_k = sex_i,
           age_bin_k = age_bin_i,
           symptom_distance(s_i, s_k) <= d_max}
```

也就是疾病不同、但 nuisance 尽可能匹配。这样模型不能靠年龄、性别、症状或来源区分正负。

### 首版 loss

使用 multi-positive supervised contrastive loss：

```text
L_dis = -mean_i log [sum_{j in P(i)} exp(sim(z_i,z_j)/tau)
                     / sum_{q in P(i) union N(i)} exp(sim(z_i,z_q)/tau)]
```

同时采用 raw-preserving residual adapter，而不是把 raw audio 表征完全投影掉：

```text
z_i = LayerNorm(r_i + alpha * g(r_i))
```

`alpha` 初始化为 0。当前 audit 中 raw 表征通常不差于 aligned 表征，因此 preservation 不是可选
装饰，而是由已有结果直接要求的设计边界。

第一版不同时加入 adversarial loss。先只测试“跨 nuisance 正样本 + nuisance-matched 负样本 +
residual preservation”这一变量；如果它失败，再单独预注册 gradient reversal／HSIC，不事后堆 loss。

## 4. 数据可行性门必须先过

在任何 GPU 训练前，先对候选数据集报告：

1. 每个 `disease × source × sex × age_bin × symptom_bin` 单元人数；
2. 每个 anchor 的 `|P(i)|`、`|N(i)|` 分布；
3. 至少 80% anchor 有一个跨来源 positive 和一个 nuisance-matched negative；
4. patient-disjoint train/validation/test；
5. train、validation 和 untouched external test 的 source--label contingency；
6. 只用 nuisance 的疾病 AUROC；
7. frozen raw audio 的疾病 AUROC。

UKCOVID Standard train 中 T+T 几乎没有阴性、REACT 几乎没有阳性，可能无法构造合法的
跨来源 disease positives／matched negatives。若第 3 条不过，UKCOVID 不能承担该方法训练；
不能降低门槛强行开跑。

## 5. 正式实验 arm

| arm | 目的 |
|---|---|
| `raw audio` | 不训练对齐的基线 |
| `correct metadata alignment` | 当前 RespiraMFM-style 参照 |
| `within-label shuffle` | 当前 population-association 控制 |
| `naive same-label SupCon` | 证明“同标签正样本”本身是否仍学习 cohort |
| `cross-stratum invariant SupCon` | 新 positive/negative 设计 |
| `raw + invariant residual` | 主方法，保证原始声学信息不被投影删除 |

AST-6L 与 OPERA-CT 使用同一协议；至少五个 seed。所有超参数只在 source-domain validation
冻结。matched／external test 不用于挑 loss、温度、stratum 距离或 residual weight。

## 6. 评价不能只看疾病 AUROC

需要同时报告三轴，而不是合成一个分数：

### Disease transfer

- untouched external/matched AUROC；
- source-calibrated NLL、Brier、calibration slope；
- participant-disjoint target calibration sensitivity；
- 相对 `raw audio` 和 `naive same-label SupCon` 的配对 CI。

### Correspondence／invariance

- disease-label retrieval；
- unique patient-profile retrieval；
- source、sex、age、symptom、recording artefact probes；
- `same disease / cross source` 与 `same source / cross disease` 的检索分解。

### Information preservation

- `raw + invariant` 相对 `raw` 的 CKA／线性可解码性；
- audible event probe（只有真实可靠标签时）；
- raw 分支被消融后性能是否下降。

## 7. 事前成功标准

主方法只有同时满足以下条件才叫成功：

1. 在 untouched matched/external test 上，相对 raw audio 的 paired ΔAUROC 或 Δ(−NLL) CI
   排除 0，且方向有利；
2. AST 与 OPERA 方向一致，不能只挑一个 backbone；
3. 相对 naive same-label SupCon，source／sex／age profile retrieval 或 probe 明显下降；
4. disease retrieval 与 disease transfer 同时不下降；
5. correct 配对、naive same-label 和 invariant 三者使用同样训练预算；
6. source-domain 提升而 external 不提升，判为失败，不能改写成“学到了疾病结构”。

如果只减少 nuisance probe 但疾病 transfer 不升，只能说成功移除了可解码 nuisance；如果疾病
AUROC 升但 nuisance 不降，则不能归因于 invariance。

## 8. 与当前 ICASSP 稿件的边界

当前 ICASSP 稿件已经形成完整 audit 闭环，不应为了增加一个方法结果而重新打开 test-set-driven
迭代。最稳妥顺序是：

1. 先提交／完成 audit paper；
2. 在新外部数据上完成上述可行性门；
3. 单独冻结 mitigation 预注册；
4. 才运行新 positive-pair 方法。

因此目前仓库中的准确状态是：**audit 完成；mitigation 有设计、无实验。**
