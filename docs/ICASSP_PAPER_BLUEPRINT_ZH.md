# ICASSP 2027 论文蓝图（冻结主线）

日期：2026-08-20（Day 1 与 Day 2--4 后更新）

## 暂定题目

**Does Clinical Metadata Alignment Transfer Disease Evidence? A Pairing-Controlled
Audit of Respiratory Audio under Cohort Shift**

中文直译：临床 metadata 对齐是否真的传递疾病证据？——呼吸音在队列偏移下的配对控制审计。

## 一句话故事

在 UKCOVID 的训练分布中，招募来源几乎等于 COVID 标签，因此音频与症状／人口学
metadata 的对齐既可能学习真实的个人对应，也可能学习队列结构。我们用同标签内打乱
作为关键控制：它保留标签级共现，却破坏逐患者对应。正确配对确实比同标签打乱多保留了
个人对应信息，但该增量没有转化为协变量平衡人群中的可迁移疾病收益。

Coswara 外部患者级压力测试在冻结的数据平衡门上 NO-GO，没有运行模型。作为补救，稿件
增加了 matched direct-fusion attribution control 和完整的 OPERA-CT 第二编码器稳健性；
它们增强内部机制归因，但不能把 UKCOVID 发现改写成外部确认或普遍规律。

## 唯一研究问题

> 当疾病标签与队列结构高度混淆时，audio--metadata alignment 学到的是能够跨人群迁移的
> 疾病证据，还是只在源数据分布中有用的患者／队列对应？

不再同时讲 audible reasoning、局部 grounding、schema 对自然语言、teacher 失败或
H1--H13 的探索历史。它们属于仓库审计记录，不属于这篇四页论文的主线。

## 三项贡献

1. **配对控制协议。** 引入 `within-label shuffle`：在相同疾病标签内打乱 metadata，
   保留疾病与 metadata 的总体共现，只移除逐患者对应；再用 `global shuffle` 测量完全
   破坏对应后的下界。
2. **协变量平衡的迁移审计。** 同时报告源域和 matched 人群，并把校准负对数似然作为
   主要指标，避免只看源域 AUROC 得出“对齐有效”的结论。
3. **机制与稳健性分解。** direct fusion 表明 matched null 不全是 alignment 特有；probe
   表明正确配对相对同标签打乱保留性别与部分年龄；OPERA-CT 重复了源域正向、matched
   校准负向的主方向。

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

## 四页结构

### 第 1 页：问题与设计

- 临床 metadata 不一定能从声音中听出，却可能与标签强烈共现；
- UKCOVID 的 `source -> label` 极端混淆；
- 图 1：correct / within-label / global 三种配对及 source-to-matched shift；
- 三项贡献。

### 第 2 页：方法

- 数据、患者级划分与外部确认；
- 冻结 audio／text encoder，仅训练 projector；
- 三种配对、下游同一分类头与校准协议；
- 配对 bootstrap、主次指标和 one-shot 规则。

### 第 3 页：主要结果

- 图 2：Standard、matched、matched-long（以及外部 stress test）的 paired forest plot；
- 表 1：raw audio、correct、within-label、global 的绝对 AUROC／NLL；
- 重点陈述：源域对应信号与 matched 迁移结果分离。

### 第 4 页：机制、限制与结论

- 图 3：matched probe 的 `correct - within-label` 增量；
- 外部压力测试及其与 discovery 的异同；
- 限制：单一 alignment 架构／Stage-1 风格复现、线性 readout、数据标签质量；
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
