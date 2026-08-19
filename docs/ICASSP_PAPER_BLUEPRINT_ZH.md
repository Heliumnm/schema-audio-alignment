# ICASSP 2027 论文蓝图（冻结主线）

日期：2026-08-19

## 暂定题目

**Does Clinical Metadata Alignment Transfer Disease Evidence? A Pairing-Controlled
Audit of Respiratory Audio under Cohort Shift**

中文直译：临床 metadata 对齐是否真的传递疾病证据？——呼吸音在队列偏移下的配对控制审计。

## 一句话故事

在 UKCOVID 的训练分布中，招募来源几乎等于 COVID 标签，因此音频与症状／人口学
metadata 的对齐既可能学习真实的个人对应，也可能学习队列结构。我们用同标签内打乱
作为关键控制：它保留标签级共现，却破坏逐患者对应。正确配对确实比同标签打乱多保留了
个人对应信息，但该增量没有转化为协变量平衡人群中的可迁移疾病收益。

外部确认集必须独立报告；若其主要比较不复现，论文不得把 UKCOVID 发现写成确认性结论。

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
3. **机制分解。** probe 表明正确配对不是“什么都没学到”：它相对同标签打乱稳定保留
   性别、部分保留年龄；但这些个人对应没有变成 matched 人群中的疾病增益。

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

以上全部叫 **discovery audit**，因为 UKCOVID 官方测试集曾参与早期协议设计。

## 外部确认必须回答的比较

外部数据必须先冻结患者级 train／validation／confirmation manifest，然后运行：

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

## 确认结果的判读规则

### 若外部结果复现 UKCOVID 的方向

可以写：在两个数据来源中，正确 metadata 配对能够学习个人对应，但没有观察到该对应
带来可迁移疾病收益；这支持在临床 audio--metadata 对齐研究中常规加入同标签打乱和
患者级分布偏移评测。

### 若外部结果显示正确对齐有明确迁移收益

不能隐藏或改门槛。论文应改写为边界条件研究：UKCOVID 的极端招募混淆使对齐失败，
但在外部数据中对齐可以迁移；重点比较何种数据条件决定收益。

### 若外部数据无法形成可靠患者级确认集

不把现有 UKCOVID discovery audit 伪装成确认结果。主会投稿降级为 workshop／短文，
或等待合适外部数据。内部 residual-long 已因无法复刻官方十岁年龄分层而被提前否决。

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

- 图 2：Standard、matched、matched-long（以及外部 confirmation）的 paired forest plot；
- 表 1：raw audio、correct、within-label、global 的绝对 AUROC／NLL；
- 重点陈述：源域对应信号与 matched 迁移结果分离。

### 第 4 页：机制、限制与结论

- 图 3：matched probe 的 `correct - within-label` 增量；
- 外部确认结果及其与 discovery 的异同；
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

## 投稿门槛

在 2026-09-16 前，主会稿至少需要：

1. 一个满足患者身份、标签来源、固定录音类型和患者级无泄漏划分的外部数据集；
2. 在读取 confirmation 指标前提交 manifest、hash、matching／split 和主要比较；
3. 外部数据上的 artifacts、raw audio、metadata、direct fusion 与三个配对 alignment arms；
4. discovery 与 confirmation 分开报告，不根据 confirmation 结果移动门槛；
5. 代码与逐参与者逐 seed 预测可审计。

