# Transfer Repair v2 外部数据 pair-support gate 预注册

日期：2026-09-04  
状态：**在读取任何 Repair v2 embedding、prediction 或 matched-target model score 前冻结。**

## 1. 目的与固定顺序

UKCOVID 已因 source--label 结构性空单元而在 Repair v2 数据门停止。本轮只回答：CODA TB 和
Cambridge 的 source train 是否包含足够的跨环境同病 positive，以及同环境、异病、协变量可比的
negative，从而支持同一个 cross-cohort disease contrastive objective。

数据集顺序事前固定为：

1. **CODA TB 为 primary**；环境定义为 `country`；
2. **Cambridge Task 2 为 secondary**；环境定义为 `platform`。

不得根据模型表现交换 primary。两套数据门均可运行，但正式训练优先级仍按上述顺序。

## 2. Model blindness

每套数据门只读取各自冻结 manifest 中 `splits == train` 的 participant ID、疾病标签、环境和下述
匹配协变量。不得打开：

- 音频波形或 AST／OPERA／HeAR embedding；
- Correct／Within／Global／Repair projector 或 checkpoint；
- profile retrieval、疾病 AUROC、NLL、Brier 或 calibration；
- validation、source-test、matched-target 的模型预测。

公开结果只保存 aggregate counts、比例、距离分布、文件哈希和脚本哈希，不保存 participant ID。

## 3. 共同 candidate 定义

对 source-train anchor `i`：

```text
P(i) = 同疾病标签、不同环境、不同 participant 的训练参与者
N(i) = 相反疾病标签、同环境、满足数据集特定 nuisance matching 的训练参与者
```

每个 eligible anchor 必须同时拥有至少 2 个 `P(i)` 和至少 1 个 `N(i)`。缺失值是显式类别，
不得作为通配符。candidate 只能来自同一个 source train，self-pairing 为 0。

## 4. CODA TB 的冻结规则

- 疾病：TB reference label；
- 环境：`country`；
- positive：同 TB、不同 country；
- negative：异 TB、同 country、同性别、年龄差不超过 **10 年**；
- clinical Hamming distance 不超过 **2**，字段固定为：`tb_prior`、`hemoptysis`、
  `weight_loss`、`smoke_lweek`、`fever`、`night_sweats`、`hiv_status`。

年龄 10 年对应 CODA 已冻结 matching cost 中的一个 age unit；七个 clinical 字段每个不同值记
1，缺失与非缺失也记 1。该规则在运行 source-train coverage 前固定，不根据结果改变。

## 5. Cambridge 的冻结规则

- 疾病：COVID label；
- 环境：`platform`；
- positive：同 COVID、不同 platform；
- negative：异 COVID、同 platform，并在现有 matched endpoint 的 exact fields 上全部相同：
  `age_band`、`sex`、`cough`、`fever`、`sore_throat`、`shortness_of_breath`、
  `asthma`、`other_respiratory`。

`smoker` 不进入 strict eligibility，因为既有 Cambridge match-first 协议把它作为 pair cost／fine
balance 字段而非 exact field；本门会额外报告 strict negative 中 smoker 完全相同的比例。

## 6. 统一 GO／NO-GO 判据

每个数据集必须同时满足：

1. source train 总体 joint coverage（positive 与 negative 同时合格）至少 **80%**；
2. 两个疾病标签各自 joint coverage 均至少 **80%**；
3. 每个在 source train 出现的环境都同时包含两个疾病标签；
4. 所有 candidate 来自 source train，target participant 使用数为 0，self-pairing 数为 0；
5. manifest 中 participant 唯一，source train 同时包含两个疾病标签和至少两个环境。

任何一项失败即为 `NO_GO`。禁止通过移除环境条件、放宽 80%、纳入 validation／matched target、
或根据模型结果修改 nuisance matching 来救回。

## 7. 必须报告的 aggregate-only 输出

- source train N 与疾病标签人数；
- 每个 `domain × label` 的人数；
- cross-domain same-label positive coverage；
- within-domain opposite-label strict-negative coverage；
- joint coverage（总体与逐标签）；
- candidate count 与最近 nuisance distance／caliper 分布；
- target leakage 和 self-pairing 检查；
- manifest 与 gate script 的 SHA-256。

## 8. 数据门后的动作

- CODA `GO`：先做 CODA × AST 的 S0／S1，再按单独冻结的正式训练协议运行；
- CODA `NO_GO`、Cambridge `GO`：Cambridge 可作为透明标记的 fallback primary，但必须保留 CODA
  `NO_GO`，不能写成事前首选；
- 两者都 `NO_GO`：停止 Repair v2，不生成新表示或预测；
- 任一数据集正式 Stage 1 失败：不使用 matched target 调 lambda、pair 数或 matching rule。

本文件只冻结数据可行性；它不自动授权正式模型训练。正式 sampler、loss、readout、成功门槛与
测试解锁顺序必须在数据门 `GO` 后另写预注册并经过核对。
