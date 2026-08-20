# ICASSP 2027 论文蓝图（Route A：Audit Framework）

日期：2026-08-20（Day 1 与 Day 2--4 后更新）

## 暂定题目

**Pairing-Controlled Auditing of Clinical Audio--Metadata Alignment: Separating
Participant Correspondence from Transferable Disease Evidence**

中文直译：临床音频--metadata 对齐的配对控制审计——分离患者对应与可迁移疾病证据。

## 一句话故事

现有临床 audio--metadata alignment 的评价把三种不同能力混在一起：学习逐患者
correspondence、学习标签／人群层面的统计共现，以及学习可跨人群迁移的疾病证据。我们提出
Pairing-Controlled Transfer Audit，用 correct／within-label／global 三种配对和 matched
evaluation 把三者分开。正确配对可以学到真实患者对应并增加源域信心，但这不保证在协变量
平衡人群中获得疾病排序收益。

Coswara 外部患者级压力测试在冻结的数据平衡门上 NO-GO，没有运行模型。作为补救，稿件
增加了 matched direct-fusion attribution control 和完整的 OPERA-CT 第二编码器稳健性；
它们增强内部机制归因，但不能把 UKCOVID 发现改写成外部确认或普遍规律。

## 核心 conceptual contribution

传统问题只有一句：“alignment 是否提高 downstream accuracy？”本稿把它拆成三个可分别
证伪的问题：

1. alignment 是否学会了真实的 metadata profile correspondence？
2. 学到的是逐患者信息，还是仅有疾病标签／群体共现？
3. 这些 correspondence 是否转化成协变量平衡人群中的疾病排序与可靠概率？

不再同时讲 audible reasoning、局部 grounding、schema 对自然语言、teacher 失败或
H1--H13 的探索历史。它们属于仓库审计记录，不属于这篇四页论文的主线。

## 三层贡献

1. **问题贡献。** 指出 correspondence learning、label-level association 与 transferable
   clinical evidence 是三个不同能力，现有 downstream-only 评价无法区分。
2. **方法贡献。** 提出 Pairing-Controlled Transfer Audit：`correct-within` 测逐样本对应，
   `within-global` 测标签级共现，profile retrieval 验证优化是否真的学会对应，taxonomy
   probes 说明对应内容，matched evaluation 与 probability-transport audit 检验迁移。
3. **科学发现。** 两个呼吸音 backbone 上，correct pairing 学到了真实患者 context 并产生
   源域排序／信心增益，但这份增益没有稳定变成 matched disease ranking；受控 synthetic
   stress test 只在通过预注册准入门时用于展示一种充分机制。

## Introduction 的固定四段逻辑

1. **领域假设：** 更好的 audio--metadata alignment 被默认等价于更好的临床表示；
2. **评价漏洞：** metadata 同时包含 patient、clinical-context 与 cohort/protocol 信息，
   contrastive loss 只要求 positive pair 接近，不知道哪部分可迁移；
3. **研究问题：** 如何区分 correspondence success 与 portable disease evidence？
4. **解决方案：** Pairing-Controlled Transfer Audit，随后给出 UKCOVID discovery、两个
   backbone 和 controlled synthetic mechanism 的证据链。

## UKCOVID discovery 结果（固定）

- 数据混淆：`recruitment source -> COVID` 在 Standard train 的 AUROC 为 0.9966，
  在 matched test 为 0.5000。
- 源域个人对应信号：Standard 上 `correct - within-label` 的 ΔAUROC 为
  +0.0140，95% CI [+0.0040, +0.0235]。
- 预注册主要结果：matched 上 `correct - within-label` 的 Δ(−NLL) 为
  −0.0249，95% CI [−0.0443, −0.0069]；五个 seed 全部为负。
- matched 上 ΔAUROC 为 +0.0062，95% CI [−0.0144, +0.0279]，没有可检出的排序增益。
- matched-long 方向一致：Δ(−NLL) −0.0214，95% CI [−0.0330, −0.0101]。
- 机制 probe：matched 上 `correct - within-label` 的性别 AUROC 增量为
  +0.1912 [+0.1677, +0.2156]；年龄 ≥65 为 +0.0465 [+0.0069, +0.0854]，
  但 normalized 年龄敏感性区间覆盖 0。症状与招募来源没有稳定的个人级增量。
- matched direct fusion：`metadata + raw AST − metadata only` 的 ΔAUROC 为
  +0.0028 [−0.0019,+0.0071]，Δ(−NLL) 为 −0.0034 [−0.0194,+0.0126]；原始音频在
  metadata 之上没有可检出的线性增量。
- OPERA-CT：Standard `correct − within-label` 的 ΔAUROC 为
  +0.0156 [+0.0082,+0.0230]；matched Δ(−NLL) 为
  −0.0176 [−0.0335,−0.0010]，五个 seed 全负；matched AUROC 为
  −0.0020 [−0.0191,+0.0168]；matched-long NLL 同方向。

以上全部叫 **discovery audit**，因为 UKCOVID 官方测试集曾参与早期协议设计。

## Coswara 外部压力测试的冻结结局

Zenodo v1.0 的 2,746 条 `cough-heavy.wav` 已全部提取并审计，2,646 条通过 objective QC。
冻结匹配最多 112 对；在 100 对下限时最大类别 SMD 为 0.140，超过预注册 0.12，因此数据
门判为 NO-GO。没有读取任何 Coswara 模型分数；以下比较保留为未来获得足够重叠外部数据
时的预注册模板。

## 未来外部压力测试必须回答的比较

外部数据必须先冻结患者级 train／validation／evaluation manifest，然后运行：

- recording artefacts only；
- frozen raw audio representation；
- metadata only；
- metadata + raw audio（直接拼接）；
- correct / within-label / global 三个 alignment arm；
- metadata + 各 alignment representation（同一个直接拼接分类头）。

主要比较：

1. alignment：`correct - within-label` 的 Δ(−NLL)；
2. fusion：`metadata + correct - metadata + within-label` 的 Δ(−NLL)；
3. direct audio increment：`metadata + raw audio - metadata only`；
4. alignment versus raw：`metadata + correct - metadata + raw audio`。

AUROC、Brier score 和 calibration slope 为次要指标。所有比较在同一参与者、同一 seed
上配对，并同时重采样参与者与 seed。

## 外部压力测试的判读规则

### 若外部结果与 UKCOVID 同方向且区间排除 0

可以写：Coswara 这个匹配队列也观察到相同方向；这支持在临床 audio--metadata 对齐
研究中常规加入同标签打乱和患者级分布偏移评测。仍不能写成跨数据集的普遍无效证明。

### 若外部结果与 UKCOVID 同方向但区间覆盖 0

只能写：方向一致，但外部估计不精确；不能把“没有显著差异”写成等效或确认。

### 若外部结果显示正确对齐有明确迁移收益

不能隐藏或改门槛。论文应改写为边界条件研究：UKCOVID 的极端招募混淆使对齐失败，
但在外部数据中对齐可以迁移；重点比较何种数据条件决定收益。

### 若外部结果方向相反且区间覆盖 0

写成外部结果不一致且不精确，不挑选其他指标把方向改回来。

### 若外部数据无法形成至少 100 个合格匹配对

按预注册判 Coswara formal gate 为 NO-GO，不更换录音类型或匹配阈值。主会稿只剩
UKCOVID discovery audit，必须同时补第二个 backbone／readout 稳健性，或降级为
workshop／短文。内部 residual-long 已因无法复刻官方十岁年龄分层而被提前否决。

## 实验叙事顺序（冻结）

1. **Does alignment learn correspondence?** 正式 unique-profile retrieval：MRR、R@1、R@10，
   correct／within／global 五 seed 配对；
2. **What correspondence is learned?** information-channel matrix：录音采集、患者背景、
   clinical context、cohort/protocol 与 disease；
3. **Does correspondence transfer?** Standard→matched／matched-long 的 AUROC、NLL、
   metadata-only／raw／aligned／fusion；这是 punchline；
4. **Why can this separation occur?** 只有通过准入门的 D+S synthetic confounding sweep；
5. **Robustness, not a second story.** OPERA、raw-preserving direct fusion、固定 MLP 和
   probability-transport audit 用紧凑表格／一句话封住替代解释。

## 四页结构

### 第 1 页：评价歧义与 audit framework

- 第一段直接提出 alignment 的隐含假设；
- 图 1 不以 encoder architecture 为视觉中心，而画 alignment ambiguity：同一个 positive pair
  可能承载 patient／context／cohort／disease 多种通道；
- 图 1 下半部分接 correct／within-label／global 的 pairing decomposition；
- 三层贡献：问题、协议、failure mode。

### 第 2 页：方法与 correspondence 操作检验

- 数据、患者级划分与 discovery 定位；
- 冻结 audio／text encoder，仅训练 projector；
- 三种配对、unique-profile retrieval、probe taxonomy；
- metadata-only、raw audio、aligned audio 与 direct fusion；
- 配对 bootstrap、主次指标和 one-shot 规则。

### 第 3 页：从 correspondence 到 transfer

- 先报 retrieval `Correct > Within > Global` 是否成立；
- 紧凑 taxonomy matrix 回答学到了什么；
- 主图使用 Correspondence--Transfer Map 或 pairing forest：对应学会了，但 matched disease
  gain 没有跟上；
- 主表必须含 metadata only、raw audio、aligned audio、raw+metadata，避免看起来遗漏最强
  临床基线。

### 第 4 页：机制、稳健性与限制

- synthetic 只有通过预注册正文门才占一张双面板图：confounding-strength heatmap +
  source→matched shift curve；未过门则完全不进正文；
- OPERA、MLP、raw-preserving 与 probability transport 用一段压缩；
- Coswara NO-GO 压成限制中的一句，不再占完整 Results 小节；
- 限制：没有外部患者级确认、单一 alignment 架构／Stage-1 风格复现、数据标签质量；
- 结论：把同标签打乱与协变量平衡迁移评测作为 audio--metadata alignment 的最低审计要求。

## 禁止的过度表述

- 不说“metadata alignment 普遍无效”；
- 不说“AST 没有疾病信息”；
- 不说“对齐让 AST 编码了更多人口学信息”——raw AST 的绝对 probe 反而更高；
- 不说“完整复现或推翻 RespiraMFM”——这里只复现其 Stage-1 风格的表示对齐；
- 不把 CI 覆盖 0 写成“证明没有信息”；
- 不把 UKCOVID discovery 结果叫独立确认；
- 不把残余 Standard-long 叫未预测 holdout；
- 不用 source-domain AUROC 提升代替 matched 迁移证据。

## 当前投稿口径

Coswara 未达到原先的外部门槛，因此当前 ICASSP 主会稿只能明确标为单数据集 discovery
audit。已完成的 direct fusion 与第二 backbone 解决了“alignment 特有”和“单一编码器”
两个主要归因缺口，但不等于外部确认。投稿前必须做到：

1. 把 UKCOVID、Coswara NO-GO、direct fusion 与 OPERA-CT 全部按冻结口径报告；
2. 明确限制为一个 Stage-1 projector、线性 readout 和极端混淆数据集；
3. 不声称跨数据集确认或 metadata alignment 普遍有害；
4. 保留代码、manifest、hash 与逐参与者逐 seed 预测审计；
5. 若截止前获得真正未触碰且有足够协变量重叠的外部集，只能按独立预注册追加，不能用来
   改写已经冻结的 UKCOVID 判据。
