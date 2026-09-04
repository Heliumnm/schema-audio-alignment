# Transfer Repair v1：源域 nuisance 子空间擦除 + raw skip

日期：2026-09-04

状态：**探索性、训练前冻结。尚未读取本实验的 matched／matched-long 分数。**

这不是新的大型音频模型训练，也不改写已经完成的 audit。它是一个低成本的 post-hoc
repair：保留冻结的 raw audio embedding 和已经训练完的 500-epoch Correct projector，只学习
若干线性 nuisance 方向，并在疾病 readout 之前把这些方向从 Correct 表示中投影掉。

## 1. 问题与边界

现有结果表明，Correct metadata alignment 能提高逐参与者 metadata correspondence，但没有稳定
提高 covariate-balanced disease transfer。Repair v1 只回答：

> 在不重训音频编码器和 projector 的前提下，删除 Correct 表示里可线性解码的患者／cohort／
> 录音属性方向，并保留 raw audio skip，能否改善 matched population 上的疾病迁移？

本实验已经受到既有 matched 结果启发，所以无论结果如何都属于 exploratory proof-of-concept，
不能称为 untouched confirmation。若为正，仍需在新外部数据上预注册确认。

## 2. 绝对不动的部分

- 冻结 AST-6L、OPERA-CT、HeAR audio embeddings；
- 冻结 Correct metadata projectors 及其 epoch-500 checkpoints；
- 不覆盖或继续训练任何既有 checkpoint；
- Standard train 只用于拟合 nuisance 擦除器和疾病／probe readout；
- 完整 Standard validation 只用于既有 one-SE `C` 选择和 Platt calibration；
- matched 与 matched-long 不用于选择 nuisance、rank、readout 或任何超参数；
- 疾病 readout、校准和 participant × seed bootstrap 沿用既有评估代码。

## 3. Primary nuisance set

首版只使用以下七个、在看到 Repair 结果前固定的二元目标：

1. `recruitment_source`；
2. `sex_female`；
3. `age_65plus`；
4. `long_recording`；
5. `large_file`；
6. `loud_recording`；
7. `clipped`。

症状不进入 primary eraser。症状既可能是 cohort proxy，也可能包含真实疾病信息；在没有新数据区分
两者前，把症状一起删除会让结果无法归因。症状擦除只能在之后作为单独预注册的 sensitivity。

## 4. 擦除算法

对每个 backbone 和 Correct projector seed 独立处理。

1. 只在 Standard train 上拟合 `StandardScaler`，得到标准化 Correct 表示 `C`；
2. 对每个 nuisance target，固定用 `LogisticRegression(C=1.0, max_iter=5000)`，无 class
   weighting、无超参数搜索；缺失标签仅从该 target 的拟合行中排除；
3. 取七个分类器的 coefficient direction，各自单位归一化后堆成矩阵 `W`；
4. 对 `W` 做 float64 SVD，rank 阈值固定为 `1e-8 * largest_singular_value`；
5. 用其行空间的正交基 `Q` 做一次性投影：

```text
C_erased = C - (C Q^T) Q
```

推理时不需要患者 metadata；只需保存 train-fitted scaler 与 `Q`。本方法只删除至多七个线性方向，
不声称消除了所有 nuisance，也不称为因果去混淆或 exact LEACE。

主表示保留 raw skip：

```text
R_primary = concat(raw_audio, C_erased)
```

## 5. Frozen arms

| arm | 含义 |
|---|---|
| `raw` | 原始冻结音频表示 |
| `correct` | 原 Correct metadata-aligned 表示 |
| `within_label` | 原 within-label shuffled 对照，仅作既有参照 |
| `raw_correct` | raw 与未修复 Correct 直接拼接 |
| `correct_erased` | 只保留擦除后的 Correct 分支 |
| `raw_correct_erased` | **主 Repair arm**：raw skip + 擦除后的 Correct |

## 6. 分阶段解锁

### S0：纯合成 self-test

- 确认 projection 的输出有限、非零、确定性；
- 确认擦除后与 `Q` 正交；
- 确认合成 nuisance 的 held-out 线性可解码性下降；
- 不读取任何真实数据。

### S1：source-only technical smoke

只读取 Standard train／validation 指标，不读取 matched 或 matched-long 分数。技术门为：

- 五个 seed 全部完成，输入／输出 hash 与 eraser rank 被保存；
- 所有预测有限，原 checkpoint 未被改写；
- `correct_erased` 相对 `correct` 的七项 validation nuisance AUROC 宏平均下降；
- 至少 5/7 项不增加超过 0.01；
- 疾病 validation 只作运行诊断，不设 GO 数字，也不据此修改方法。

S1 不过则停止；通过后才允许显式执行 `--score-tests`。

### S2：一次性 exploratory matched scoring

主比较：

```text
raw_correct_erased - raw_correct
```

主 endpoint：matched 上 paired `Delta AUROC`。关键次要 endpoint：

- matched paired `Delta(-NLL)`、Brier 与 calibration slope；
- `raw_correct_erased - raw`，确认 repair 没有靠丢掉全部 alignment 退化为 raw；
- `correct_erased - correct` 的 nuisance probes；
- matched-long 同方向 sensitivity。

## 7. 成功与失败的措辞

单个 backbone 的 exploratory success 必须同时满足：

1. 主比较 matched `Delta AUROC >= +0.01`，paired 95% CI 下界大于 0；
2. `raw_correct_erased` 不劣于 raw；
3. nuisance probe 宏平均下降，且不能由单一 target 驱动；
4. matched `Delta(-NLL)` 不显著变差。

进入论文主方法结果还额外要求：至少两个 backbone 同方向，并在至少一个此前未参与设计的外部数据集
上确认。只降低 nuisance probe 而不提高 disease transfer，只能写“移除了部分线性可解码 nuisance”；
source validation 提升而 matched 不提升，判为 repair 失败。

## 8. 与后续大训练的关系

Repair v1 是低成本筛查。只有它提供一致的 target-transfer 信号，才值得启动
`DISEASE_INVARIANT_ALIGNMENT_FUTURE_PLAN_ZH.md` 中的大训练：跨 nuisance positive、
nuisance-matched negative 和 residual adapter。Repair v1 失败时不通过事后增加 nuisance 字段、
rank 或非线性 eraser 来继续搜索 matched test。
