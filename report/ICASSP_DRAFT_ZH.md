# 对齐成功不等于疾病迁移：临床音频—患者信息对齐的配对控制审计

英文暂定题目：

> **Successful Clinical Audio--Metadata Alignment Does Not Imply Disease Transfer: A Pairing-Controlled Audit**

版本：中文论文母稿，2026-09-11

用途：用于继续压缩成 ICASSP 四页正文和英文稿。

证据状态：UKCOVID 为 discovery audit；CODA TB、Cambridge 和 Coswara 为不同等级的外部敏感性分析，不冒充 untouched confirmation。

---

## 摘要

临床音频模型越来越多地利用年龄、性别、症状和既往病史等患者信息，并通过对比学习将音频表示与对应的临床文本对齐。这类方法通常把对齐成功视为疾病表示得到改善的证据。然而，患者信息同时混合了疾病关联、人口学背景、症状以及人群构成；对比损失只要求模型找对配对，并不知道哪些信息能够跨人群迁移。本文提出 **Pairing-Controlled Transfer Audit**，用正确配对（Correct）、同标签内换人（Within-label）和全局换人（Global）分离逐患者 profile correspondence 与标签／人群共现，再结合 profile retrieval、信息通道 probe、原始音频参照和协变量平衡后的疾病评测，检验 correspondence 是否真正转化为 transferable disease evidence。

在 UKCOVID 中，招募来源在训练集上几乎可以直接预测 COVID（AUROC 0.9966），而在 matched population 中降为 0.5000。AST-6L 和 OPERA-CT 的 Correct 配对均显著提高 profile retrieval，证明 alignment 的确学到了患者资料对应关系；Correct 相对 Within-label 明显保留性别信息（matched probe ΔAUROC 分别为 +0.1912 和 +0.1125），而 COVID 增量很小且不确定（+0.0062 和 −0.0020）。这种 correspondence 没有形成跨音频 backbone、跨线性／非线性 readout、且优于 raw audio 的稳定疾病迁移收益。目标域重新校准可以消除主要的 NLL 差距，却不会创造 AUROC 收益，说明额外误差主要来自目标人群不支持的置信度，而非已经证实的排序破坏。CODA TB、Cambridge COVID-19 Sounds 和 Coswara 的二次／事后敏感性分析重复了相同形状：患者 profile 与性别 correspondence 经常增强，而协变量平衡后的 COVID/TB 收益方向不一致且置信区间跨零。结果表明，临床音频—患者信息研究必须分别验证“是否学会配对”“学到了什么”以及“能否跨人群迁移”，不能用 alignment loss、检索成功或源域 AUROC 代替疾病迁移证据。

**关键词：** clinical audio；audio--metadata alignment；contrastive learning；confounding；domain shift

---

## 1. Introduction

咳嗽、呼吸和肺部听诊音频被广泛用于呼吸疾病识别。近年来，研究者除了训练纯音频模型，也开始把患者年龄、性别、症状、吸烟史和既往疾病作为辅助信息。AST、OPERA 和 HeAR 等模型提供通用的音频表示；BTS、RespLLM 等系统进一步融合音频与临床背景；RespiraMFM 则明确使用对比学习，将同一患者的呼吸音频和临床文本拉近，再把对齐后的音频表示用于下游预测 [@gong2021ast; @zhang2024opera; @siam2026respiramfm]。

这类设计隐含了一个看似自然的推理：

```text
音频与患者信息对齐得更好
        ↓
音频表示包含更多临床知识
        ↓
疾病预测和跨人群泛化更好
```

问题在于，临床 metadata 并不是单一语义来源。它可能同时描述：

- 与疾病相关的症状和病史；
- 年龄、性别和生活方式等患者背景；
- 与入组方式、医院、平台或采集流程有关的人群信息；
- 缺失模式等间接的 cohort proxy。

对比学习本身无法判断其中哪部分是“真正应该从声音中学习并迁移的疾病证据”。它只知道正确的 audio--metadata pair 应该比其他 pair 更接近。因此，只要年龄、性别或症状有助于区别配对，模型就有动力保留这些信息，即使它们在目标人群中不再预测疾病。

这带来一个被现有下游准确率评价掩盖的歧义：

> **Successful alignment does not necessarily imply a transferable disease representation.**

本文不把问题简化为“模型学到了疾病还是患者”的二选一，而是询问：正确配对带来的增量中，有多少来自逐患者的 profile correspondence，有多少来自标签／人群统计，又有多少能够在协变量平衡后继续改善疾病预测？

围绕这一问题，我们提出三个研究问题：

- **RQ1：** 音频—metadata alignment 是否真的使用了逐患者正确配对？
- **RQ2：** 正确配对相对同标签内换人，选择性保留了哪些信息通道？
- **RQ3：** 这些 correspondence 是否形成跨人群、跨 backbone、跨 readout 且优于 raw audio 的疾病迁移收益？

本文的贡献有三点：

1. **概念贡献。** 将 correspondence success、label/population association 和 transferable disease evidence 明确区分，指出三者不能由同一个源域 AUROC 替代。
2. **评价方法贡献。** 提出 Pairing-Controlled Transfer Audit，通过 Correct、Within-label 和 Global 三个只改变配对关系的训练臂，结合 profile retrieval、信息通道 probe、raw reference、matched transfer 和 calibration transport 形成闭环审计。
3. **实证贡献。** 在 UKCOVID 的两个主要音频 backbone 以及 CODA TB、Cambridge、Coswara 的多 backbone 敏感性分析中，正确配对反复提高患者 profile correspondence，尤其保留性别信息；但没有形成稳定、跨设置且优于 raw audio 的协变量平衡疾病收益。

我们的结论不是 metadata alignment 普遍无效或有害，而是：**配对学习是否成功与疾病证据是否迁移，必须被分别测量。**

---

## 2. Related Work

### 2.1 Respiratory audio representations

纯音频研究尝试从咳嗽、呼吸声和肺部听诊音中学习健康相关表示。Audio Spectrogram Transformer（AST）将梅尔频谱图切分为 patch，并使用 Transformer 编码 [@gong2021ast]。OPERA 进一步在多个呼吸声数据集上进行开放式呼吸声表征预训练 [@zhang2024opera]。HeAR 则提供面向健康声学任务的通用音频表示。这些工作主要回答“声音中是否包含可用于健康任务的信息”，但不直接回答把患者背景对齐进音频表示后，新增的究竟是哪类信息。

### 2.2 Audio and clinical-context fusion

BTS、RespLLM 等方法把人口学信息、症状或病史与呼吸音频共同用于预测。这类方法的主要问题是“加入患者背景是否改善任务表现”。它们与本文关注的显式表示对齐不同：融合模型允许两种模态各自提供证据，而对比 alignment 会直接要求音频表示接近患者文本空间。

既有工作也曾使用 metadata 监督呼吸音频对比学习，说明患者信息可以成为表征学习信号。RespiraMFM 更明确地采用两阶段框架：第一阶段冻结音频和文本编码器，只训练 audio-side projector，使每段音频接近对应临床文本；第二阶段再把对齐表示用于多模态预测 [@siam2026respiramfm]。本文复现和审计的是其 **Stage-1-style representation alignment**，而不是其完整的 Phi-2 + LoRA 下游系统，因此本文结果不能被解释为对 RespiraMFM 完整系统准确率的直接复现或否定。

### 2.3 Cohort confounding in clinical audio

临床音频数据往往来自不同招募渠道、设备、地点和症状人群。UKCOVID 的既有研究表明，未经调整的音频分类性能在对年龄、性别和症状等因素进行匹配后会显著下降，简单症状模型也可能达到或超过音频模型 [@coppock2024audio]。这说明源域高分可能反映 cohort construction，而不完全是可迁移的病理声音。

现有工作通常通过下游分类结果评价对齐是否有效，却较少先验证模型是否学到精确配对，再区分这种对应来自患者背景、标签共现还是疾病证据。本文填补的是这一 **evaluation gap**，而不是提出新的音频 encoder 或追求新的 leaderboard 分数。

### 2.4 Positioning of this work

本文与纯音频、简单融合和显式对齐三类工作的关系如下：

| 范式 | 代表方向 | 主要问题 |
|---|---|---|
| Audio-only representation | AST、OPERA、HeAR | 声音能否编码健康相关信息？ |
| Audio + metadata fusion | BTS、RespLLM | 患者背景能否提高预测？ |
| Audio--metadata alignment | metadata-supervised contrastive learning、RespiraMFM | 临床文本能否改善音频表示？ |
| **本文** | Pairing-Controlled Transfer Audit | 正确 alignment 学到了什么，这些信息能否迁移？ |

---

## 3. Data

### 3.1 Dataset overview

本文以 UKCOVID 为 discovery dataset，并在 CODA TB、Cambridge COVID-19 Sounds Task-2 和 Coswara 上进行不同等级的外部敏感性分析。所有建模和重采样均以 participant 为单位，避免同一参与者跨 split。

| 数据集 | 疾病／任务 | 冻结 cohort | matched target | 本文地位 |
|---|---|---:|---:|---|
| UKCOVID | PCR-referenced COVID-19 | 72,458 名有可用 cough 的参与者 | 1,814；matched-long 4,196 | discovery audit |
| CODA TB | 微生物学参考标准 TB | 1,081 名参与者 | 200 人／100 对 | match-first v2 secondary sensitivity |
| Cambridge Task-2 | custom strict COVID endpoint | 989 名参与者 | 200 人／100 对 | reconstructed external sensitivity |
| Coswara | COVID-19 | 1,701 名参与者 | 200 人／100 对 | post-hoc external stress test |

表中的 matched target 只平衡预先测量并纳入 matching 的协变量，不能声称消除了所有未测量混淆。

### 3.2 UKCOVID

UKCOVID 是大规模、PCR-referenced 的英国声音数据集 [@budd2024ukcovid]。我们固定使用 cough 录音作为主要音频。可用队列包含 72,458 名参与者；Standard train、validation 和 test 分别用于 projector 训练、source-domain readout／calibration 选择和源分布评测。官方 rebalanced matched test 包含 1,814 人，matched-long 包含 4,196 人，且与 Standard train/validation participant-disjoint。

UKCOVID 特别适合本审计，因为其训练分布存在极强的招募来源—标签耦合：recruitment source 单独预测 COVID 的 AUROC 在 Standard train 为 0.9966，而在 matched test 中为 0.5000。训练集中 `symptom_none` 在阴性中的比例为 0.786，在阳性中仅为 0.019，因此即使不把 recruitment source 明文提供给模型，症状也可以成为其强 proxy。

UKCOVID 官方 test 曾参与本项目早期协议诊断，因此最终 UKCOVID 结果被透明标记为 discovery/exploratory，而非 untouched confirmation。

### 3.3 CODA TB

CODA TB 提供来自七个国家的 solicited cough、临床 metadata 和微生物学 TB reference standard。客观 QC 后共有 1,081 名 eligible participant。原 split-first v1 数据门未满足平衡门槛；在未查看模型输出的前提下，我们另行冻结 match-first v2：先从全部 eligible participants 确定 matched target，再划分余下 development cohort。最终得到 train 603、validation 122、source-test 156 和 matched target 200 人。matched target 最大绝对 SMD 为 0.0819，最大分类水平差为 0.040。

这一结果属于 CODA Challenge Train release 内部的 secondary sensitivity analysis，不是 hidden challenge validation。

### 3.4 Cambridge COVID-19 Sounds Task-2

Cambridge 数据使用官方 Task-2 音频子集，并结合全量 metadata 重建 cough-only strict-COVID endpoint。participant namespace 审计修正了早期把 18 个 Web submission 合并为一个 participant 的错误。修正后队列包含 989 人，固定划分为 train 551、validation 117、source-test 121 和 matched target 200 人。matched target 中预先选择的协变量达到完全平衡。

由于官方 Task-2 split 文件不可用，本分析使用自定义、事前冻结的重建 endpoint 和 split，因此不是官方 benchmark replication，也不是完全独立于设计过程的确认实验。

### 3.5 Coswara

Coswara 是带有症状和 COVID 信息的呼吸声音数据集 [@bhattacharya2023coswara]。原预注册贪心 matching gate 在 100 对下得到最大 SMD 0.140，高于 0.12 门槛，因而维持 NO-GO。之后在仍未查看任何模型分数时，另行冻结全局约束 matching sensitivity，得到 train 1,041、validation 217、source-test 243 和 matched target 200 人。该 target 的年龄 SMD 为 0，最大分类 SMD 为 0.1034，最大层级比例差为 0.07。

由于它是在原数据门失败后建立的 secondary cohort，本文只把 Coswara 作为 post-hoc stress test。

### 3.6 Metadata schema and leakage exclusions

UKCOVID 的主要对齐文本采用固定的结构化 schema，包含：

- 年龄；
- 性别；
- 吸烟状态；
- 哮喘和其他呼吸系统病史；
- 预先冻结的症状字段；
- 显式 `[MISSING]` 值。

三个外部数据集使用各自预注册中可获得的对应患者字段，而不是强行补齐 UKCOVID 不存在的
字段；所有数据集共同遵守下面的 target／domain 泄漏排除原则。

文本**不包含**疾病标签、检测结果、病毒载量、招募来源、国家／site、平台、设备、时间戳、文件大小、录音时长、响度、削顶属性或 LLM 生成内容。因而本文讨论的是患者背景和症状通过统计相关性代理疾病／cohort 结构，不是把目标标签或 source 字段直接泄漏给模型。

UKCOVID train 的 20,714 名参与者对应 5,437 个不同 schema profile，最常见 profile 占 7.9%。检索和损失解释均显式考虑 profile 重复，不能把结果称为 participant identity recognition。

### 3.7 Privacy and reproducibility

患者音频、逐患者 metadata、manifest、embedding、prediction 和 checkpoint 保存在受控存储中，不提交 GitHub。仓库只公开 aggregate JSON、配置、代码、commit hash 和 SHA-256，以支持数字审计而不暴露参与者信息。

---

## 4. Method

### 4.1 Problem formulation

设第 \(i\) 名参与者具有音频 \(a_i\)、metadata 文本 \(m_i\)、疾病标签 \(y_i\) 和环境／人群变量 \(e_i\)。冻结音频编码器 \(g\) 和文本编码器 \(h\)，得到：

\[
x_i=g(a_i), \qquad t_i=h(m_i).
\]

我们只训练 audio-side projector \(f_\theta\)，使投影音频 \(f_\theta(x_i)\) 接近指定配对文本。核心实验只改变 \(a_i\) 与 \(m_j\) 的配对规则，其余架构、初始化、batch 顺序、优化器、训练步数和评测流程保持一致。

### 4.2 Three pairing-controlled arms

#### Correct

```text
audio_i ↔ metadata_i
```

保留逐参与者精确 profile correspondence，也保留源数据中的标签／人群统计关联。

#### Within-label

```text
audio_i ↔ metadata_j,
y_i = y_j,
i ≠ j
```

在同一疾病标签内固定换人。它破坏精确 participant-to-profile correspondence，同时在期望意义上保留同标签群体的 metadata 分布。

#### Global

```text
audio_i ↔ metadata_k,
i ≠ k
```

全局固定换人，破坏系统性的逐患者和标签条件配对。它仍可能偶然落到相同标签或相同 profile，因此是“无系统对应参照”，而不是数学上保证每一对都完全不同。

我们定义三个主要对比：

\[
\Delta_{C-W}=M(C)-M(W),
\]

用于衡量同标签统计之外，精确 participant/profile pairing 带来的增量；

\[
\Delta_{W-G}=M(W)-M(G),
\]

用于描述标签条件／人群共现；以及

\[
\Delta_{C-R}=M(C)-M(R),
\]

用于判断 alignment 是否真正优于 frozen raw audio representation。

这里的 \(C-W\) 不是“纯身份效应”或因果 estimand；它是控制同标签群体统计后，精确配对额外购买到的 profile correspondence。

### 4.3 Encoders and projector

UKCOVID 的主要确认轴使用两个预先冻结的音频 backbone：AST 前六层（AST-6L）和 OPERA-CT。HeAR 作为后续／外部 backbone robustness 分析。所有音频均按各 backbone 的冻结规范预处理，并在 participant 内先聚合为单一表示。

文本侧使用冻结的 Phi-2。每种唯一 schema 只编码一次，并使用 mask-aware mean pooling；UKCOVID schema 长度为 188--200 tokens，因此固定 `max_length=200`，避免原实现的 125-token 上限截断全部文本。

Projector 忠实采用 RespiraMFM Stage-1 代码的总体形式：

```text
audio dimension → 1024 → 2560
Linear + LayerNorm + ReLU + Dropout(0.1)
Linear + LayerNorm + ReLU
```

末层 ReLU 使 projector 输出非负。我们保留这一实现性质，以审计代表性 Stage-1 alignment，而不把它解释为标准对称 CLIP。

### 4.4 Contrastive objective

音频与文本向量进行 L2 归一化：

\[
z_i^a=\frac{f_\theta(x_i)}{\|f_\theta(x_i)\|},
\qquad
z_i^t=\frac{t_i}{\|t_i\|}.
\]

使用单向 audio-to-text、batch-negative InfoNCE：

\[
\mathcal{L}_{\mathrm{align}}
=-
\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(z_i^a\cdot z_i^t/\tau)}
{\sum_{j=1}^{N}\exp(z_i^a\cdot z_j^t/\tau)},
\]

其中 \(\tau=0.07\)。这不是标准 CLIP 的双向 loss。

重复 schema 会形成不可消除的 loss floor。我们实测 exact-text multi-positive 与 single-positive 的梯度等价：最大梯度差为 \(6.3\times10^{-8}\)，梯度余弦为 1.000000。因此 exact duplicate multi-positive 只改变常数项，不作为正式训练臂。near-neighbour schema 是否应作为 multi-positive 是另一个尚未检验的假设。

### 4.5 Training protocol

三个 pairing arm 统一使用：

- 5 个随机种子：0--4；
- batch size 64，`drop_last=False`；
- Adam，learning rate \(10^{-3}\)，无 scheduler 和 weight decay；
- 500 epochs；
- 只使用最后一个 checkpoint，不根据疾病 validation 选择 epoch；
- 同一 seed 内共享初始化、audio batch order 和 dropout stream；
- shuffled pairing 每 seed 生成一次，并在 500 epochs 内固定。

音频和文本编码器始终冻结，只更新 projector。正式训练前的 smoke test 只检查配对、梯度、冻结状态、数值有限性、无坍缩和可复现性，不根据三个 arm 的相对表现决定是否继续。

### 4.6 Audit endpoints

#### 4.6.1 Profile correspondence

在 validation population 上执行 audio-to-profile retrieval。由于多个参与者可能共享完全相同的 schema，我们不把某一个具体人当成唯一正确答案，而是把相同 profile 视为同一语义目标，并报告 macro-profile mean reciprocal rank（MRR），防止高频 profile 主导均值。

Retrieval 的作用是 manipulation check：它证明 alignment 是否真的使用了 pairing，不是临床 endpoint。

#### 4.6.2 Information-channel probes

在冻结表示上使用相同的正则化 probe，分别解码：

- patient：sex、age；
- clinical context：症状、既往疾病；
- cohort/acquisition：source、platform、录音属性；
- disease：COVID 或 TB。

Probe 只说明某种信息可以从表示中读出，不证明疾病分类器在因果意义上使用了它。

#### 4.6.3 Disease transfer

疾病 readout 只在 source train 训练，超参数和 source-domain calibration 只用 source validation 冻结。随后一次性评测：

- Standard/source-test：保留原 cohort structure；
- matched target：平衡预先测量的主要协变量；
- UKCOVID matched-long：participant-disjoint sensitivity population。

Matched data 不用于选择 projector、training epoch、classifier architecture 或超参数。

主报告指标包括 AUROC、negative log-likelihood（NLL）和 Brier score。所有 arm 保存逐 participant、逐 seed 预测，以便做 paired comparison。

#### 4.6.4 Raw and fusion controls

为避免把“对齐臂之间有差异”误写成“对齐优于原始声音”，我们同时评测：

- metadata-only；
- frozen raw audio；
- aligned audio；
- metadata + raw audio；
- metadata + Correct／Within／Global aligned audio；
- recording artifacts-only。

线性 readout 是主要比较，预先固定的 MLP 用于检查结果是否只是线性可读出限制。

#### 4.6.5 Probability transport

我们分别检查：

1. source-domain Platt calibration 直接迁移到 matched；
2. 只修正疾病先验；
3. 固定 prediction shrinkage；
4. 用 participant-disjoint matched-long 拟合一维 target calibrator，再原样应用到 matched。

该分析区分“疾病排序发生变化”和“源域置信度在目标人群中失配”。

### 4.7 Statistical analysis

所有关键比较均在相同 participant 和相同 seed 上配对。置信区间使用 participant × seed 分层 bootstrap；外部 matched targets 按冻结 matched pair／participant 单位重采样。我们报告观测配对差和 95% CI，不把“CI 包含零”解释成已证明零效应，也不在结果后改变等价区间或成功门槛。

---

## 5. Results

### 5.1 Source-domain performance is highly confounded

UKCOVID 的 training distribution 中，recruitment source 与 COVID 标签近乎确定对应，而 matched target 对该变量实现平衡。

| Population | Recruitment source → COVID AUROC | 解释 |
|---|---:|---|
| Standard train | 0.9966 | source 几乎等于标签 |
| Standard validation | 0.9984 | 同样高度耦合 |
| Matched test | 0.5000 | source 被平衡 |
| Matched-long | 0.5000 | source 被平衡 |

Metadata-only GBM 在 Standard test 的 AUROC 为 0.910，在 matched test 为 0.609，在 matched-long 为 0.569。原始 AST 表示在 Standard test 为约 0.708，在 matched 为约 0.538。只使用时长、RMS、削顶和文件大小等录音属性，在 matched 上也可达到约 0.534。

这些数字不能进行 AUROC 的“成分拆分”，但说明疾病评测必须在相同参与者上直接比较模型，而不能只和 0.5 比较或把 Standard 高分解释为病理声音。

### 5.2 Correct pairing learns profile correspondence

表 2 报告 validation 上 macro-profile retrieval 的 Correct−Within MRR。

| Dataset | Backbone | Correct−Within MRR | 95% CI | 判读 |
|---|---|---:|---:|---|
| UKCOVID | AST-6L | **+0.0032** | [0.0016, 0.0049] | correspondence established |
| UKCOVID | OPERA-CT | **+0.0032** | [0.0015, 0.0050] | correspondence established |
| CODA TB | AST-6L | **+0.0667** | [0.0349, 0.1021] | established |
| CODA TB | OPERA-CT | **+0.0302** | [0.0064, 0.0555] | established |
| CODA TB | HeAR | **+0.0646** | [0.0320, 0.1014] | established |
| Cambridge | AST-6L | **+0.0433** | [0.0151, 0.0799] | established |
| Cambridge | OPERA-CT | +0.0298 | [−0.0011, 0.0638] | directionally positive |
| Cambridge | HeAR | **+0.0457** | [0.0203, 0.0763] | established |
| Coswara | AST-6L | **+0.0183** | [0.0028, 0.0365] | established |
| Coswara | OPERA-CT | **+0.0243** | [0.0050, 0.0452] | established |
| Coswara | HeAR | **+0.0406** | [0.0226, 0.0619] | established |

UKCOVID 的绝对 MRR 较低，是因为 validation 中有 1,871 个候选 profile 且大量 profile 重复；配对差异仍在两个 backbone、五个 seeds 中保持正向。外部数据中，除 Cambridge OPERA 的区间略跨零外，其余均清楚表明 Correct 比 Within 更容易恢复匹配的患者 profile。

因此，后续疾病 transfer 不显著不能归因于“projector 什么也没有学到”。

Global arm 还说明仅保留同标签群体统计本身也会产生信号。以 UKCOVID AST 为例，在 Standard
test 上，Within−Global 的 COVID、cough 和 recruitment-source probe ΔAUROC 分别为
**+0.0646、+0.0758 和 +0.0647**，均为正且区间排除零；进入 matched 后，COVID 的
Within−Global 降为 +0.0126 且区间跨零，而 cough 仍为 +0.0803。这个结果说明 source
distribution 中 label-conditioned metadata association 确实可被学习，但它不等同于可迁移的
疾病声音。

### 5.3 The most reproducible retained channel is participant sex

matched population 上的 probe 揭示了 Correct pairing 相对 Within-label 主要保留什么。

| Dataset | Backbone | Sex ΔAUROC C−W | Disease ΔAUROC C−W |
|---|---|---:|---:|
| UKCOVID | AST-6L | **+0.1912** [0.1669, 0.2151] | +0.0062 [−0.0150, 0.0273] |
| UKCOVID | OPERA-CT | **+0.1125** [0.0943, 0.1311] | −0.0020 [−0.0196, 0.0158] |
| CODA TB | AST-6L | **+0.1058** [0.0582, 0.1536] | +0.0341 [−0.0221, 0.0916] |
| CODA TB | OPERA-CT | **+0.0446** [0.0058, 0.0852] | −0.0256 [−0.0872, 0.0346] |
| CODA TB | HeAR | **+0.0364** [0.0149, 0.0603] | −0.0127 [−0.0916, 0.0631] |
| Cambridge | AST-6L | **+0.1318** [0.0745, 0.1922] | −0.0448 [−0.1181, 0.0300] |
| Cambridge | OPERA-CT | **+0.1233** [0.0716, 0.1740] | +0.0077 [−0.0869, 0.1132] |
| Cambridge | HeAR | **+0.1282** [0.0893, 0.1703] | −0.0512 [−0.1186, 0.0161] |
| Coswara | AST-6L | **+0.1324** [0.0511, 0.2246] | +0.0028 [−0.0450, 0.0464] |
| Coswara | OPERA-CT | **+0.1075** [0.0469, 0.1699] | −0.0115 [−0.0750, 0.0444] |
| Coswara | HeAR | **+0.0570** [0.0189, 0.0990] | +0.0037 [−0.0646, 0.0724] |

性别是唯一在四个数据设置和绝大多数 backbone 上稳定为正且区间排除零的通道。年龄在 UKCOVID AST 和 CODA AST／OPERA 上明确，在其他设置中依赖 backbone 或统计功效。招募来源、症状和录音属性的 C−W 增量也不具同样稳定性。

这支持“Correct pairing 选择性保留 participant-associated profile information”，但不能写成“模型识别了患者身份”。同时，raw audio 的人口学信息绝对可解码性通常更高；例如 UKCOVID raw AST 在 matched 上解码性别约为 0.871、年龄约为 0.742。因此 alignment 并非凭空创造这些特征，而是相对 Within projector 更强地保存它们。

### 5.4 Correspondence does not yield robust matched disease transfer

上表最后一列给出了每个设置的 matched disease C−W。其共同形状是：

- UKCOVID 两个 backbone 的增量接近零且方向不同；
- CODA TB 的 AST 点估计为正，OPERA 和 HeAR 为负，三个区间都跨零；
- Cambridge 的三个点估计方向不一致，三个区间都跨零；
- Coswara 三个点估计接近零，区间都跨零。

因此没有跨数据集、跨 backbone 的稳定正向 disease-transfer evidence。外部 100 对 targets 的区间较宽，所以准确结论是 **inconclusive / no robust positive evidence**，而不是已经证明效应严格等于零。

Correct 相对 raw audio 同样没有清楚优势。CODA matched 上 C−R 分别为 AST +0.0078、OPERA −0.0082、HeAR −0.0550，区间均包含零；Cambridge 和 Coswara 也没有稳定 C−R 优势。换言之，alignment 学到的 profile correspondence 没有稳定超过未经 metadata 对齐的音频表示。

### 5.5 Stronger readout and direct fusion do not rescue the main claim

在 UKCOVID matched 上，raw audio 加入 metadata 的 ΔAUROC 为 AST +0.0028 [−0.0016, 0.0073]、OPERA +0.0041 [−0.0006, 0.0087]，均不显著。

在线性 fusion 中，`metadata + Correct` 相对 `metadata + Within` 的 ΔAUROC 为：

- AST：+0.0156 [0.0034, 0.0276]；
- OPERA：+0.0021 [−0.0068, 0.0107]。

AST 出现小的 ranking signal，但 OPERA 没有复现；两者 NLL 都明显变差，而且 `metadata + Correct` 没有优于 `metadata + raw audio`。

使用预先固定的 MLP 后，audio-only C−W 为 AST +0.0050 [−0.0154, 0.0270]、OPERA −0.0052 [−0.0208, 0.0110]；C−Raw 分别为 −0.0113 [−0.0291, 0.0068] 和 −0.0247 [−0.0404, −0.0081]。更强非线性读出没有恢复跨 backbone 的疾病收益。

Cambridge 的 direct fusion 中只有 AST 出现 AUROC 增量，OPERA 和 HeAR 未复现，NLL 也不提供跨 backbone 支持。因而个别正向点估计不构成稳健方法收益。

### 5.6 Calibration analysis identifies unsupported confidence

UKCOVID 中，将 source calibrator 直接迁移到 matched 后：

| Backbone | C−W ΔAUROC | C−W Δ(−NLL) |
|---|---:|---:|
| AST-6L | +0.0062 | −0.0249 |
| OPERA-CT | −0.0020 | −0.0176 |

负的 Δ(−NLL) 表明 Correct 的概率质量更差。只修正目标患病率不能消除差距；预测越尖锐，AST 的 NLL regret 越大。

但使用 participant-disjoint matched-long 拟合 target calibrator，再迁移到 matched 后：

| Backbone | Target-calibrated C−W ΔAUROC | Target-calibrated Δ(−NLL) |
|---|---:|---:|
| AST-6L | +0.0062 [−0.0156, 0.0264] | +0.00013 [−0.00227, 0.00243] |
| OPERA-CT | −0.0020 [−0.0194, 0.0164] | −0.00020 [−0.00261, 0.00228] |

重新校准几乎完全消除了 NLL 差距，但不会改变 AUROC。说明原来的 NLL 惩罚主要是 source cohort 学到的置信度在目标人群中没有支持，而不是已经证明 Correct 破坏了疾病排序。

### 5.7 Synthetic mechanism test did not establish a general law

受控 synthetic 实验覆盖 3 个 confounding strength × 3 个 shortcut strength × 10 seeds，共 90 次模拟。Objective 能够通过 correspondence gate，但预注册的 high-confounding、dose-trend 和 rho=0 boundary gates 均未通过。因此 synthetic 只能证明对比目标有能力学习 correspondence，不能证明“confounding 越强，正确 alignment 必然越差”这一一般机制。

该结果放入补充材料和限制，不作为正文正向机制证据。

### 5.8 Repair feasibility stopped before model training

后续 Repair v2 尝试要求每个 anchor 同时拥有：跨环境同标签 positive，以及同环境、异标签且 nuisance 可比的 negative。模型盲 pair-support gate 的 joint eligible coverage 为：

| Dataset | Joint coverage | 冻结门槛 | 结果 |
|---|---:|---:|---|
| UKCOVID | 0.246% | 80% | NO-GO |
| CODA TB | 67.50% | 80% | NO-GO |
| Cambridge | 45.37% | 80% | NO-GO |

因此 Repair v2 没有训练任何模型。这是单数据集内缺少必要 pair support 的数据可行性结果，不是 repair 方法效果失败。

---

## 6. Discussion

### 6.1 Main finding

本文最重要的结果不是“alignment 没学到东西”。恰恰相反，profile retrieval 证明模型反复利用了正确 audio--metadata pairing。真正的问题是，它最稳定保留的是患者 profile，尤其是性别，而不是可稳定迁移的疾病排序。

说得更直白一些：

> 把咳嗽和本人的资料卡放在一起训练，模型确实更会判断“这段声音对应什么样的患者资料”；但把年龄、性别和症状等已测变量尽量配平以后，它没有稳定地更会判断这个人有没有 COVID 或 TB。

这正是 correspondence success 与 disease transfer 之间的评价歧义。

### 6.2 Why source-domain gains are insufficient

在 UKCOVID，招募来源在 source train 中几乎等于标签；在 CODA，部分 source-test 上 Correct 的点估计也优于 Within，但进入 matched target 后方向发生变化。只报告源分布 AUROC 会把 population association、participant correspondence 和 disease evidence 混在一起。

因此，论文不能仅凭以下现象声称疾病表示改善：

- alignment loss 下降；
- matched pair retrieval 提高；
- embedding 更可分；
- source-test AUROC 增加。

这些现象最多证明对齐目标被优化。临床价值仍需在预先定义、participant-disjoint、协变量平衡的 target population 中验证，并与 raw audio 和 metadata-only 基线直接配对比较。

### 6.3 What the three pairing arms contribute

Global shuffling 同时破坏逐患者配对和标签条件关联，因此单独比较 Correct 与 Global 是有歧义的。Within-label 是审计的关键：它保留同标签人群的 metadata 统计，同时破坏精确参与者配对。

因此：

```text
Correct > Within retrieval
    → 精确 profile pairing 被利用

Within > Global
    → 标签／人群统计本身可被利用

Correct > Within on matched disease
且 Correct > Raw
    → 才可能支持可迁移疾病收益
```

这一分解是本文相对仅报告下游准确率的主要方法贡献。

### 6.4 Implications for clinical audio-language research

临床音频—文本系统至少应报告：

1. within-label pairing control；
2. global pairing control；
3. duplicate-aware profile retrieval，确认 alignment 是否发生；
4. raw audio 和 metadata-only reference；
5. patient、clinical、cohort/acquisition 和 disease probes；
6. participant-disjoint、covariate-balanced transfer evaluation；
7. AUROC 之外的 NLL／Brier 与 calibration transport。

缺少这些控制时，研究者很容易把“模型能找回患者资料”误写成“模型从声音中学到了可迁移疾病证据”。

### 6.5 Implications for future model design

本文没有验证一个成功的修复方法。下一步可以把 metadata 拆成三类：

- 可听或可能与病理声音一致的证据；
- 仅作为临床上下文的患者信息；
- cohort、platform、device 和 recording nuisance。

对比学习只负责对齐“同一声学事实的两种表达”；年龄、症状、病史等不可直接从声音可靠恢复的信息，应在最终预测阶段 late fusion，而不应强迫音频 encoder 重建。另一条方向是构造跨环境同疾病 positives 和环境内异疾病 hard negatives，使目标从 instance-level correspondence 转向 disease-level invariance。

不过 Repair v2 的数据门表明，当前三个单数据集都缺少足够 joint pair support。这意味着真正的修复更可能需要多 cohort 联合训练，而不是继续在 UKCOVID 内部调整 loss 或 matching 门槛。该方向属于后续方法论文，不纳入本文实证主张。

---

## 7. Limitations

第一，UKCOVID 是 discovery audit。其官方 test 曾参与早期协议诊断，所以本文不把它描述为 untouched confirmation。

第二，外部结果的证据等级不同。CODA 是公开 training release 内的 match-first v2 secondary analysis；Cambridge 使用自定义 strict-COVID endpoint 和重建 split；Coswara 是原 gate 失败后的 post-hoc stress test。它们提高了现象的一致性，但不等同于一个完全未触碰的外部确认。

第三，matched evaluation 只平衡已测量协变量，不能消除未测混淆、selection bias 或 measurement error。

第四，probe 证明信息可解码，不证明疾病分类器因果使用该信息；profile retrieval 证明 pairing manipulation 成功，不证明临床价值。

第五，本文审计的是冻结 encoder 加单一 Stage-1-style projector。虽然使用了 AST、OPERA 和 HeAR，并检查线性和固定 MLP readout，但未覆盖所有 alignment architecture、loss 或 end-to-end fine-tuning，也没有运行 RespiraMFM 的完整 Phi-2 + LoRA Stage 2。

第六，CODA、Cambridge 和 Coswara 的 matched target 均为 100 对，置信区间较宽。CI 包含零意味着当前数据未建立稳健正向证据，不等于证明真实效应为零。

第七，synthetic 实验没有通过一般机制门，Repair v2 也停在数据支持门。本文因此提出的是评价框架和实证 failure pattern，而不是已经被因果证明的普遍规律或已经验证的新修复模型。

---

## 8. Conclusion

本文提出 Pairing-Controlled Transfer Audit，用 Correct、Within-label 和 Global 三种配对控制，将患者 profile correspondence、标签／人群共现和可迁移疾病证据分开评价。

在 UKCOVID 中，AST-6L 和 OPERA-CT 都明确学到了逐患者 profile correspondence，Correct pairing 尤其保留性别信息；但 matched COVID 增量很小、不具跨 backbone 和跨 readout 稳健性，也没有稳定超过 raw audio。目标域校准消除了主要 NLL 差距，却没有创造疾病排序收益。CODA TB、Cambridge 和 Coswara 的 secondary/post-hoc analyses 进一步重复了“correspondence 明确、性别信息增强、matched disease transfer 不确定”的形状。

因此，临床音频—metadata alignment 的评价不能停留在“是否对齐成功”或“源域 AUROC 是否提高”。最低限度的证据链应同时回答：

```text
模型是否真的使用了配对？
模型通过配对学到了什么？
这些信息能否在不同患者构成中迁移？
它是否优于未经对齐的原始音频？
```

本文的核心结论是：

> **Correct audio--metadata pairing learned reproducible patient-profile correspondence, but this correspondence did not yield a robust cross-backbone improvement in covariate-balanced disease transfer.**

---

## References used by this draft

正式投稿时由 `paper/references.bib` 统一排版。当前正文已经使用或需要补齐的核心文献包括：

- Gong et al. Audio Spectrogram Transformer. Interspeech 2021. `gong2021ast`.
- Zhang et al. Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking. NeurIPS 2024. `zhang2024opera`.
- Baur et al. HeAR: Health Acoustic Representations. 2024. **待补 BibTeX。**
- Moummad et al. Pretraining Respiratory Sound Representations using Metadata and Contrastive Learning. 2022. **待补 BibTeX。**
- Kim et al. BTS: Audio, metadata and text for respiratory sound classification. 2024. **待核对并补 BibTeX。**
- Zhang et al. RespLLM. 2025. **待核对并补 BibTeX。**
- Siam et al. RespiraMFM. ACL 2026. `siam2026respiramfm`.
- Budd et al. A Large-Scale and PCR-Referenced Vocal Audio Dataset for COVID-19. Scientific Data 2024. `budd2024ukcovid`.
- Coppock et al. Audio-Based AI Classifiers Show No Evidence of Improved COVID-19 Screening over Simple Symptoms Checkers. Nature Machine Intelligence 2024. `coppock2024audio`.
- Bhattacharya et al. Coswara. Scientific Data 2023. `bhattacharya2023coswara`.
- CODA TB dataset paper. **待补 BibTeX。**
- Cambridge COVID-19 Sounds dataset／benchmark paper. **待补 BibTeX。**

---

## Internal evidence map（投稿时删除）

- UKCOVID 主结果与 claim 边界：`docs/ICASSP_EXECUTION_STATUS_ZH.md`
- UKCOVID 可审计汇总：`results/route_a_final/`
- Metadata alignment：`docs/METADATA_ALIGNMENT_OUTCOME_ZH.md`
- CODA TB：`docs/CODA_TB_FORMAL_RESULTS_ZH.md`
- Cambridge：`docs/CAMBRIDGE_TASK2_FORMAL_RESULTS_ZH.md`
- Coswara：`docs/COSWARA_FORMAL_RESULTS_ZH.md`
- Synthetic：`docs/SYNTHETIC_CORRESPONDENCE_TRANSFER_PREREG_ZH.md`
- Repair v2 数据门：`docs/TRANSFER_REPAIR_V2_EXTERNAL_GATE_OUTCOME_ZH.md`
- 下一阶段方法（未执行）：`docs/DISEASE_INVARIANT_ALIGNMENT_FUTURE_PLAN_ZH.md`
