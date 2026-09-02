# CODA TB 正式模型实验 v1：冻结协议

冻结日期：2026-09-03  
状态：**在读取任何 CODA 音频表示、projector、retrieval、probe 或 TB 模型分数之前冻结**  
数据前提：`match-first-v2` model-blind gate 已返回 `GO`

## 1. 研究问题与地位

本实验只问一个问题：

> 在 UKCOVID 之外的 TB 咳嗽数据中，把音频与同一 participant 的临床 metadata 对齐，
> 是否会学到比“同 TB 标签但不同 participant”更强的个体对应关系；如果会，这种增量是否同时
> 改善协变量平衡人群中的 TB 迁移？

它是 CODA TB **公开 training release 内部的、预注册的外部敏感性分析**。它不是 CODA
challenge 的隐藏 validation/test 结果，也不是 untouched confirmatory replication。原因是 v1
数据门失败后，v2 看过聚合数据统计才把流程改成 match-first。v1 与 v2 的历史必须一起保留。

本实验不声称构建新 TB 诊断系统，也不以超过 CODA leaderboard 为目标。它检验的是
`correspondence != transfer` 这一 audit 结论能否在另一疾病、另一采集体系和多国人群中复现。

## 2. 已冻结的数据

只使用 CODA TB DREAM Challenge Train (`syn39711065`) 的 clinic-supervised **solicited
cough**。不使用 longitudinal cough，不使用官方隐藏 validation，不增加其他音频模态。

受控输入的 SHA-256：

- participant manifest：`f3a423d6dc2f09c1ec4a21af7ff45774cf7cb25fe3d84d3ed0285461e7781a6c`
- matched-pair manifest：`f2587ffe9ac14102f73dc18c1b65186c23d38f5953be83e3fa5ac6836c43cea4`
- audio-QC manifest：`b2df1c98bdcbbf5302d12a0f34b873b87e03e7cf3db3e5d4b5b9f16180004848`

固定分析单位为 participant。1,081 名 eligible participant 和 9,772 条已通过 QC 的 solicited
cough 构成唯一队列：

| population | TB− | TB+ | total | 用途 |
|---|---:|---:|---:|---|
| train | 476 | 127 | 603 | 训练 projector 和下游 readout |
| validation | 98 | 24 | 122 | 选择 readout 正则、拟合 source calibration、做 retrieval |
| source-test | 116 | 40 | 156 | 源分布测试，只在模型与 readout 冻结后读取 |
| matched-target | 100 | 100 | 200 | 100 个冻结 matched pairs，最终迁移终点 |

`matched-target` 从不选择 encoder、文本、projector、epoch、分类器、正则、阈值或校准器。
source-test 和 matched-target 都只在全部训练与验证步骤完成后读取一次。matched-target 的
bootstrap 单位是 pair；其他 population 的单位是 participant。

## 3. 两个冻结音频编码器

两个 backbone 独立完整执行同一协议，不根据一个 backbone 的结果修改另一个。

### AST-6L

- checkpoint：`MIT/ast-finetuned-audioset-10-10-0.4593` 的本地冻结 snapshot；运行时记录
  snapshot 路径、Transformers 版本和全部权重文件 SHA-256；
- 只保留 encoder 前六层，全部参数冻结，输出维度 768；
- 16 kHz、mono，不做逐文件峰值归一化；
- 每个窗口 10.24 s；短录音右侧补零；长录音使用
  `ceil(duration / 10.24)` 个均匀起点窗口完整覆盖首尾；
- 丢弃两个 special tokens；只平均完全落在真实音频内的 12×101 patch；跨越 padding 边界的
  patch 丢弃；
- 先等权平均窗口得到 recording 表示，再等权平均 participant 的全部 retained recordings。

### OPERA-CT

- checkpoint：`encoder-operaCT.ckpt`，预期 SHA-256
  `83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae`；
- encoder 全冻结，输出维度 768；
- 沿用 OPERA 原生 16 kHz mel 前处理；先按其固定参数去除首尾静音；
- 每个窗口 8 s；长录音用 `ceil(trimmed_duration / 8)` 个均匀窗口完整覆盖；短录音沿用
  OPERA 的 repeat padding；
- 先等权平均窗口得到 recording 表示，再等权平均 participant 的全部 retained recordings。

两个 encoder 对每位 participant 都只能产生一个表示，因此 recordings 多的 participant 不会
在 contrastive loss、下游模型或 bootstrap 中获得更高权重。

### 提取前技术门

每个 backbone 先对 28 名 participant 做分层 preflight：四个 population × 两个 TB label，
并覆盖 audio-count、总时长和单文件时长的极端值。重复提取必须逐位一致；输出必须为
`28×768`、有限、非零；窗口必须覆盖首尾；participant 文件数必须与冻结 manifest 一致。

正式提取必须覆盖 1,081/1,081 participant。任何解码或前处理失败均视为技术阻塞：停止并记录，
不得静默删除 participant、替换录音或重新匹配 target。只允许修正可复现的软件错误，且修正
必须在任何下游模型分数被读取前提交。

## 4. 冻结 metadata schema 与文本轴

文本只由数据门已经冻结的 16 个 profile 字段构成，顺序固定：

`age, sex, height, weight, reported_cough_dur, tb_prior, tb_prior_pul,
tb_prior_extrapul, tb_prior_unknown, hemoptysis, heart_rate, temperature,
weight_loss, smoke_lweek, fever, night_sweats`。

明确排除：TB label、country/site、device、participant/audio ID、HIV、录音数量、时长、响度、
削顶和文件属性。country 与 HIV 可用于 v2 matching，但不进入 alignment text。

文本是固定字段标签，不生成自然语言 reasoning：

```text
[AGE=...] [SEX=...] [HEIGHT=...] ... [NIGHT_SWEATS=...]
```

数值先解析为 finite float，再用 Python `.10g` 规范化；分类值使用数据门清洗后的原值；空值写
`[MISSING]`，不得把 missing 改成 NO。发现未知字段或非有限数值即硬错误。冻结队列中 1,081
个 profile 均唯一，因此 exact-text multi-positive 在这里没有适用对象，不建立该 arm。

文本编码器为冻结 Phi-2，revision
`810d367871c1d460086d9f82db8696f2e0a0fcd0`。使用右 padding、mask-aware mean pooling、
`max_length=256`；先对全部文本计算真实 token 长度，任何一条超过 256 都停止，不允许截断后
继续。每条唯一文本只编码一次，打乱编码顺序再恢复后必须逐位一致。输出为 2560 维。

## 5. 三个 pairing arm

对 AST 和 OPERA 分别训练：

| arm | 音频 i 的文本正样本 | 能保留什么 |
|---|---|---|
| `correct` | participant i 自己的 metadata | TB group + participant correspondence |
| `within_label` | 同 TB label、不同 participant 的 metadata | TB group，破坏 participant correspondence |
| `global` | 全 train 随机不同 participant 的 metadata | 破坏 TB group 与 participant correspondence |

shuffle 是 train 内的无固定点双射；每个 seed 生成一次并在 500 epochs 中固定。不得跨 split，
不得使用 target participant。完整 pairing 与 hash 保存在受控结果目录。

正式 seeds 固定为 `0,1,2,3,4`。同一 backbone、同一 seed 内，三臂共享 projector 初始化、
epoch batch 顺序和 dropout 随机流，唯一差异是 pairing。

## 6. Projector 与 loss

复用 RespiraMFM Stage-1 的 paper-faithful 实现，来源 commit
`b4224f231f947f3ac3ba8dcafe1f11d8a9f3e525`：

- audio-side projector：`768 -> 1024 -> 2560`；
- block 1：Linear + LayerNorm + ReLU + Dropout(0.1)；
- block 2：Linear + LayerNorm + ReLU；
- audio encoder 与 Phi-2 均冻结，只训练 projector；
- 单向 audio-to-text InfoNCE，temperature `0.07`，batch 内文本为 negatives；
- batch size 64，`shuffle=True`，`drop_last=False`，每个 participant 每 epoch 恰好出现一次；
- Adam，lr `1e-3`，PyTorch 默认参数；无 weight decay、scheduler、gradient clipping；
- 500 epochs；只允许 epoch-500 checkpoint 进入正式评测，不按 loss 或验证分数挑 epoch。

## 7. 训练前 smoke tests

1. **S0 合成正确性：** 手算 InfoNCE 与实现一致；三种 pairing 合法；encoder 全冻结；resume 与
   uninterrupted 逐位一致；输出有限、非零、非负。
2. **S1 train-only rehearsal：** 每个 backbone 用 seed 0 在 train 上短跑 500 updates；只检查
   loss 有限、梯度非零、表示未坍缩和输入/配对 hash。不得读取 validation、source-test 或
   matched-target 的疾病、retrieval 或 probe 分数。

S0/S1 失败只允许修技术错误；修复必须留下提交记录并重新从 S0 开始。S1 的三臂排序不用于
决定是否运行正式实验。

## 8. 评测链与冻结 readout

### 8.1 Alignment 是否真的发生：profile retrieval

只在 validation 的 122 名 participant 上评估；candidate bank 是 validation 的 122 个唯一
profile。query 为 audio projector 输出，candidate 为冻结 Phi-2 向量，使用 L2-normalised
cosine ranking。

主量：macro-profile MRR 的 `correct - within_label`。由于 profile 全唯一，macro 与 micro
数值相同，但两者都报告。次要量：R@1、R@10 和 `within_label - global`。

CI 使用 profile/participant × seed 分层配对 bootstrap，10,000 次。只有 `correct-within` MRR
的 95% CI 下界大于 0，才认为 **participant correspondence 已建立**。

### 8.2 TB disease transfer

对 `raw`, `correct`, `within_label`, `global` 分别训练完全相同的线性 logistic readout。另报
`metadata_only`、`metadata+raw`、`metadata+correct`、`metadata+within`、`metadata+global`
作为直接融合参照；metadata 使用同一 16 字段的 train-vocabulary one-hot，显式加入
`[UNSEEN_IN_TRAIN]`，不含 label/country/HIV。

- 标准化器只在 train 拟合；
- C 网格固定为 `[0.001,0.003,0.01,0.03,0.1,0.3,1,3]`；
- 每个 candidate readout 只在 train 拟合；validation 用固定
  `StratifiedKFold(5, shuffle=True, random_state=20260903)` 估计 AUROC 均值与 SE；
- 选择均值最优 C 的 one-SE 带内最小 C；每折必须同时有 TB+ 与 TB−；
- 用 validation 的 logits 做五折 OOF Platt 只供验证诊断；最终 Platt calibrator 在全部
  validation 上拟合一次，并原样应用于 source-test 与 matched-target；
- target 内不得重新校准、选阈值或调 C。

主要 transfer estimand 是 matched-target 上 `correct - within_label` 的配对
`delta(-NLL)`；共同主判据还要求配对 `delta AUROC` 同时为正且 CI 下界大于 0。报告 Brier、
calibration slope/intercept、source-test 结果、`correct-raw`、`within-global` 以及 fusion 结果，
但不替换主判据。

matched-target CI 按 100 个 matched pair × 5 seeds 分层配对 bootstrap 10,000 次；
source-test 按 participant × seed 分层配对 bootstrap。表中报告原始样本的观测差，不能用
bootstrap 均值代替。

### 8.3 学到了什么：信息通道 probes

对 `raw/correct/within/global` 使用相同 source-only 线性 probe，主要比较 matched-target 上
`correct-within`：

- patient：sex、`age>=45`；
- clinical context：prior TB、fever、night sweats、hemoptysis、weight loss、recent smoking；
- cohort：country（7 类，macro one-vs-rest AUROC）；
- acquisition：`audio_count>=8`。

阈值 45 岁和 8 条录音在本文冻结时已由全队列分布决定，之后不改。缺失 target 显式排除，
不得把 missing 当 negative。若 train 或 validation 缺少某一类，该 probe 报
`not_estimable`，不换定义。

## 9. 事先冻结的解释分支

1. **Retrieval 不通过：** 没有建立 alignment correspondence；任何 disease 差异都不能归因于
   correct participant pairing。
2. **Retrieval 通过，matched 两个主指标都通过：** correct correspondence 与平衡 TB 迁移
   同时改善；这会修正 UKCOVID 中“correspondence 未带来 transfer”的边界，但仍需官方
   CODA hidden evaluation 或新外部数据确认。
3. **Retrieval 通过，transfer CI 全部落入等价区间：** 支持 correspondence 与 transfer
   分离。等价区间固定为 `delta AUROC +/-0.02`、`delta(-NLL) +/-0.01`。
4. **Retrieval 通过，但 transfer CI 含 0 且未完全落入等价区间：** 只能写
   `inconclusive`，不能写“没有迁移”。
5. **正确配对更差：** 如实报告 negative transfer，不用 source-test 或 fusion 的有利结果替换
   matched 主终点。

任何结论都必须同时报告 AST 与 OPERA；方向不一致时写 backbone-dependent，不挑选较好者。

## 10. 运行顺序与隐私

固定顺序：

1. 提交并推送本文档；
2. 实现代码与单元测试；
3. S0；
4. AST、OPERA 各自 preflight 与完整 raw extraction；
5. Phi-2 文本轴检查；
6. 两个 backbone 的 S1 train-only rehearsal；
7. 三臂 × 五 seeds × 500 epochs；
8. 冻结 validation readout/calibration；
9. 一次性读取 source-test 与 matched-target；
10. 生成公开 aggregate JSON 与中文结果文档。

原始音频、临床表、participant manifest、matched pairs、逐 participant embedding/prediction、
checkpoint 与 pairing 不得进入公开 GitHub。公开仓库只保存代码、配置模板、输入/输出 hash、
聚合统计、CI 和不含 ID 的日志。

