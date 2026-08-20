# ICASSP 严格补实验预注册（设计稿，尚未运行）

日期：2026-08-20  
状态：**E1/E2 只冻结设计，尚未运行；E3 已做过一次既有 AST 预测的 post-hoc 诊断，
已看数字的部分明确标出，不冒充预注册。**

## 0. 研究边界

这批实验不是为了把现有负结果“救成正结果”，而是回答当前稿件剩下的三个最强替代解释：

1. AST 的 direct-fusion null 是否只是因为 AST 音频表征太弱；
2. 疾病信息是否存在于对齐表征中，但线性分类头读不出来；
3. `correct - within-label` 的 matched NLL 差异是否只是目标患病率变化或简单的校准尺度问题。

UKCOVID 的 matched 与 matched-long 已参与前期方案修正，因此以下全部仍是
**discovery-cohort robustness analysis**，不能写成新的独立确认。Coswara 已按冻结数据门
NO-GO；不修改平衡阈值、不读取模型结果、不把小样本录音级结果冒充外部确认。

## 1. 总执行顺序

1. **E3：校准迁移分解（先做；先正式记录已知 post-hoc 结果，再跑尚未读取的跨集合
   probability-transport sensitivity）。**
2. **E1：OPERA-CT direct-fusion 与 raw-preserving fusion（必须做，线性头且成本很低）。**
3. **E4：Synthetic Correspondence--Transfer Stress Test。** 只作为受控机制实验；单独
   预注册，达到正文准入门才进入四页稿件。
4. **E2：固定小型 MLP 非线性读出（最后做，直接回答线性头限制）。**
5. 完成 E1--E4 后一次性更新论文。不开新的 audio--text alignment head，不生成 LLM
   reasoning，不更换 schema，不追加第三个 backbone。

所有 test 结果在对应模型、预测、配置和文件 hash 写盘后才统一计算。任何技术失败只能修
实现错误；不得因为 matched 指标不好而换 architecture、epoch、loss、calibrator 或阈值。
E4 的完整生成模型、参数网格、ROPE 与正文准入门见
`SYNTHETIC_CORRESPONDENCE_TRANSFER_PREREG_ZH.md`。

## 1.1 论文统一框架：Pairing-Controlled Transfer Audit

不把信息预先画成“可迁移→不可迁移”的因果阶梯。年龄、性别甚至招募来源都可能在声音和
录音流程中留下真实痕迹；是否成为 shortcut 必须由迁移评测决定。统一使用三个配对差值：

- `C-W = correct - within-label`：逐样本 metadata correspondence；
- `W-G = within-label - global`：标签级／群体级共现；
- `C-R = correct - raw`：alignment 相对原始音频表示的改变。

这些差值在五类信息目标上测量：

| 信息通道 | 冻结测量 |
|---|---|
| 录音采集 | long recording、large file、loud、clipped |
| 患者背景 | sex、age 65+ |
| clinical context | cough any、no symptoms |
| cohort/protocol | recruitment source |
| disease target | COVID AUROC／NLL |

UKCOVID 没有可靠的事件级可听病理标签，因此不能把 COVID 标签直接叫作“audible evidence”。
普通 probe 只说明可解码性，不证明疾病分类器因果使用该变量。

另外正式补一个与疾病分类无关的 correspondence manipulation check：在完整 Standard
validation 上，把重复 schema 折叠成唯一 profile candidate，以 audio projection 检索真实
profile；主指标为 profile-level MRR，R@1/R@10 为辅助。三个 alignment arms 使用完全相同
candidate bank，五 seed 配对。定义 `CG = MRR(C)-MRR(W)`。

检索实现现在固定为：

- query 是 Standard validation 的每名参与者；candidate bank 是该集合出现的所有唯一
  schema profile，每个 profile 只保留一个冻结 Phi-2 embedding；
- audio projector output 与 candidate text embedding 都 L2 normalize，用 cosine similarity；
- 只使用 raw post-ReLU projector output，不在结果后增加 normalized／last-token 版本；
- 主指标为 **macro-profile MRR**：先在每个真实 profile 内平均 query reciprocal rank，再对
  profile 等权平均，避免 7.9% 的模态 profile 主导；
- micro MRR、R@1、R@10 为辅助；
- 主要比较 `C-W`，关键次要 `C-G` 与 `W-G`；
- macro CI 按 profile × seed 配对 bootstrap，micro CI 按 participant × seed 配对 bootstrap，
  10,000 次、固定 RNG `20260820`；
- 保存每个 query 的 participant ID、真实 profile ID、rank、seed 和候选 bank hash；
- 只在 Standard validation 做正式操作检验，不因 matched retrieval 更好或更差改变疾病结论。

不构造把 MRR、probe AUROC 和 NLL 相减的单一 “ATG” 分数。论文使用
**Correspondence--Transfer Map**：横轴 `CG`，纵轴 matched disease transfer gain
`TG_AUC=AUC(C)-AUC(W)`，颜色表示 `TG_NLL=NLL(W)-NLL(C)`。不同量纲保持分开。

---

## 2. E1：跨 backbone direct fusion 与 raw-preserving control

### 2.1 问题

AST 上 `metadata + raw AST - metadata only` 在 matched 的 ΔAUROC 为 +0.0028，区间覆盖
0。但 raw OPERA-CT 的 matched AUROC 高于 raw AST（0.5772 对 0.5380）。因此必须回答：

> 一个更适合呼吸音的 raw backbone 是否在 metadata 之上包含可检出的疾病增量；如果有，
> Stage-1 alignment 是保留、增加，还是丢失了这部分增量？

### 2.2 冻结输入臂

沿用完全相同的 schema one-hot、Standard train、完整 Standard validation、one-SE 线性头、
Standard-validation Platt 校准器、五个 alignment seeds 和同一患者顺序。

对 OPERA-CT 跑：

| arm | 下游输入 |
|---|---|
| `M` | metadata only |
| `M+A` | metadata + recording artefacts |
| `M+R` | metadata + raw OPERA-CT |
| `M+C` | metadata + correct-aligned OPERA-CT |
| `M+W` | metadata + within-label-aligned OPERA-CT |
| `M+G` | metadata + global-shuffled-aligned OPERA-CT |
| `M+R+C` | metadata + raw OPERA-CT + correct alignment |
| `M+R+W` | metadata + raw OPERA-CT + within-label alignment |
| `M+R+G` | metadata + raw OPERA-CT + global alignment |

`M` 直接复用已有冻结预测，不能重新拟合后挑较好的版本。`M+A` 此前只有 validation gate，
没有正式 matched 全队列预测，因此按同一 v3 协议新拟合一次，并作为两个 backbone 共用的
确定性参照。`M+R` 也为确定性 representation，只训练一个确定性线性头并为配对 bootstrap
复制 seed 轴；alignment 臂按原五个 seed 配对。AST 只补跑此前没有的
`M+R+C / M+R+W / M+R+G`，已有臂逐位复用。

### 2.3 冻结比较层级

主要缺口比较：

1. **OPERA direct audio increment：** `M+R - M`，matched ΔAUROC。

关键次要比较：

2. `M+C - M+W`：逐人正确配对是否在 metadata 条件下增加 matched 排序／校准；
3. `M+C - M+R`：alignment 是否优于未经对齐的 raw fusion；
4. `M+R+C - M+R`：保留 raw 表征后，correct alignment 是否仍提供增量；
5. `M+R+C - M+R+W`：在 raw 信息固定后，逐人正确配对是否提供增量；
6. `M+R - M+A`：音频增量是否超过简单录音伪影。

同时用完全相同的 probe 训练／选择协议，补齐 AST 与 OPERA-CT 的
`raw / correct / within / global` information-channel matrix。主报告 `C-W`，次报告 `W-G`
与 `C-R`；不把多个 probe 事后平均成一个总分。

主要指标是 matched 的 paired ΔAUROC。Δ(−NLL)、Brier、calibration slope、matched-long 与
Standard 为次要指标。区间使用现有 participant × seed 配对层级 bootstrap；确定性臂复制
seed 轴只为对齐索引，不把五份复制当成五次独立训练。

### 2.4 事前判读

- `M+R > M`，而 `M+C` 不超过 `M+R`：说明 raw OPERA 有 metadata-conditional 疾病
  排序信息，测试的 alignment 没有保留它；这是最强的 alignment-specific 证据。
- `M+R` 与 `M+C` 都不超过 `M`：不能把 null 归咎于 alignment；源训练分布／标签与可用
  音频证据才是更上游的瓶颈。
- `M+C > M+R`：现有“alignment 没有带来 matched 增益”的结论必须改写，不能隐藏。
- `M+R+C > M+R` 且 `M+R+C > M+R+W`：只能写“保留 raw 的融合控制中，正确配对提供了
  额外线性可读信息”；若仅追平 `M+R`，只能写“skip connection 防止丢失”，不能称提升。
- CI 覆盖 0 一律写成估计不精确，不能写等效或无信息。

---

## 3. E2：固定小型 MLP 非线性读出敏感性

### 3.1 问题

现有疾病头全部是正则化 logistic regression。E2 只回答：

> 在不改变 audio/text encoder 和 alignment 的情况下，一个预先固定的小型非线性读出能否
> 从 correct representation 中恢复线性头没有读出的 matched 疾病排序？

它不测试更强 projector，也不允许在 matched 上调参。

### 3.2 冻结模型

- 输入在 Standard train 上按每列均值／标准差标准化；零方差列保持 0；
- `Linear(D,128) -> LayerNorm(128) -> GELU -> Dropout(0.20) -> Linear(128,1)`；
- 不用 BatchNorm，不加残差，不改隐藏宽度；
- `BCEWithLogitsLoss`，不加 class weight；
- AdamW，learning rate `1e-3`，weight decay `1e-4`，gradient clipping `5.0`；
- batch size `256`，最多 `100` epochs；
- 不设 scheduler，不按 matched、matched-long 或 Standard test early-stop；
- 每个 seed 的 paired arms 共享 readout 初始化、minibatch 顺序和 dropout RNG；
- 五个 seed `0,1,2,3,4`。对 aligned arm，readout seed 与同编号 alignment seed 配对；对
  deterministic raw/metadata arm，五个 seed 只反映 readout optimiser 随机性。

epoch 的选择不使用 Standard validation，而是在 Standard train 内冻结一个
`90% inner-train / 10% inner-stop` 划分，按 `label × recruitment_source` 分层。前 10 epochs
不停止，之后以 inner-stop BCE 为准，patience `15`、min-delta `1e-4`；记录最佳 epoch，随后
用相同初始化在完整 Standard train 上重新训练恰好该 epoch 数。所有 arm 共用 inner split。
完整 Standard validation 只拟合最终 Platt calibrator，不选择 architecture、learning rate、
dropout 或其他超参数。正式运行前允许一个不读取任何 test 的 smoke：输出必须有限、非恒定，
train loss 下降；技术失败即暂停并记录，不自动换配置。

### 3.3 冻结输入矩阵

两种 backbone（AST-6L、OPERA-CT）都跑：

1. audio only：`raw / correct / within-label / global`；
2. metadata fusion：`M / M+raw / M+correct / M+within / M+global`；
3. recording artefacts only 与 `M+artefacts` 作为非线性伪影参照。

### 3.4 冻结主指标

E2 保持原稿指标层级，主要比较是：

> AST audio-only `correct - within-label` 在 matched 的 paired Δ(−NLL)。

OPERA-CT 的相同比较是预声明 replication。paired ΔAUROC 是最关键的次要指标，用来回答
“非线性可读的排序信息”；它不受单调 Platt 校准影响。依次报告：

1. audio-only `correct - within-label`；
2. fusion `M+correct - M+within-label`；
3. direct fusion `M+raw - M`；
4. `correct - raw` 与 `M+correct - M+raw`；
5. Standard、matched-long、per-seed 方向和两种 backbone。

不对两种 backbone 求 pooled p-value；分别报告点估计和 CI。不得选择“更好看的”backbone
作为主结果。

### 3.5 事前判读

- 两个 backbone 都重复“Standard 正、matched AUROC 无可检出增量／NLL 负”：结论对
  readout capacity 更稳健。
- MLP matched `correct - within` ΔAUROC 的 CI 全部高于 0：线性头是重要瓶颈，论文必须
  改成“正确对应存在非线性可读的迁移信息，源校准仍可能失败”。
- 只有 fusion 正向：说明对应信息需要 clinical context 才能使用，不能再写 audio-only null。
- seed 符号冲突且 CI 宽：写成非线性结果不稳定，不能用均值救结论。
- 只有一个 backbone 正向：写成 backbone-dependent boundary condition，不跨 backbone
  平均。

---

## 4. E3：matched NLL 的 Probability-Transport Audit

### 4.1 问题

source-calibrated NLL 是当前预注册部署指标，但 matched 患病率被构造为 0.5，Standard
validation 的患病率不同。已有 AST 预测的一次 post-hoc 诊断已经表明：NLL 排序与预测
sharpness 高度一致，不能再把负向 NLL 直接说成“正确对齐损失了更多疾病信息”。E3 区分：

1. 简单 prior/intercept shift；
2. slope／置信度尺度失配；
3. 校准之后仍然存在的排序／refinement 差异。

### 4.2 已看过的 post-hoc AST 诊断（必须披露）

以下数字已被读取，只能用来修正解释和设计后续敏感性：

- matched 常数 `p=0.5` 的 NLL 为 `0.6931`；global／within／correct／raw AST 的 NLL
  依次约为 `0.739 / 0.770 / 0.795 / 0.857`；
- 相同顺序下平均 `|p-0.5|` 约为 `0.131 / 0.161 / 0.186 / 0.225`；
- 只修正到 matched 的 50% prior 后，correct−within NLL 差仍约 `+0.0274`；
- matched 内五折 target-Platt 后，correct 与 within NLL 都约 `0.6925`，差几乎为 0；
- matched-long 的同类诊断也使差异接近 0。

因此当前最精确的工作假设是：**correct pairing 学到了真实对应并增加置信度，但 matched 上
没有相应的排序收益支撑这份置信度，产生 calibration regret。** 这不是新确认结果。

### 4.3 尚未运行、现在冻结的只读审计

只读取已经冻结的逐患者 raw logits／source-Platt probabilities；不重训 projector 或疾病头。
对 AST、OPERA-CT，audio-only 与 metadata-fusion 的 raw／correct／within-label／global 均
固定报告：

- raw-logit AUROC；
- source-Platt NLL、Brier、excess NLL（相对 `log(2)`）与 Brier skill（相对 0.25）；
- 正类和负类分别的 NLL；
- calibration-in-the-large、calibration slope；
- sharpness：logit 标准差、预测熵、prior-centered 平均绝对 logit。

固定做三种 probability transport：

1. **prior-only correction**：在 source-Platt logit 上加
   `logit(pi_target) - logit(pi_source)`，其中 `pi_source` 是完整 Standard validation 的固定
   prevalence，`pi_target=0.5` 来自 matched 抽样设计；不看单个 target label；
2. **固定置信度收缩曲线**：prior correction 后计算
   `p_lambda = sigmoid(lambda * z)`，其中
   `lambda in {0, 0.1, 0.25, 0.5, 0.75, 1}`；报告完整曲线，不选择最佳 lambda；
3. **participant-disjoint target calibration transport**：在 matched-long 上为每个 arm／seed
   拟合一维 target Platt，要求 slope 非负，然后原样应用到 participant-disjoint matched；
   反向 `matched -> matched-long` 只作预声明敏感性。

第 3 项只用于“如果拥有另一个目标域校准队列”的机制分解，不代表无标签部署性能。
bootstrap 的每个 replicate 内同时重采样 calibrator 集、evaluation 集和 seed，并重新拟合
一维 calibrator；arm 和 seed 保持配对。matched 与 matched-long 患者不得交叉。

不报告可结果后选择的 ECE。正 slope 的 Platt 校准在数学上不改变排序；正式实现将 AUROC
从校准前的 rank score 计算，将 NLL/Brier 从校准后的概率计算，并断言不存在反序。这样可
避免 slope 接近 0 时 float64 sigmoid 把不同 score 压成并列值而伪造 AUROC 变化；新增并列
值、饱和值与边界拟合会逐 arm/seed 留档。固定收缩曲线中的 `lambda=0` 是预先声明的常数
端点，不适用该排序不变性。第 2 项不得从曲线中挑一个 lambda 作为“新正式模型”。

### 4.4 判读

- prior correction 后负向 Δ(−NLL) 消失：原差异主要是患病率/intercept shift；论文必须
  降低“alignment-specific calibration failure”的强度。
- prior correction 后差距随 lambda 单调放大：额外 NLL 主要是 unsupported confidence；
- matched-long→matched Platt 后差距消失：主要是置信度 slope／尺度失配，而不是排序损失；
- target transport 后仍负，同时 ΔAUROC 也负：才支持 correct arm 的可恢复判别信息更差；
- target transport 后 correct 反而更好：必须写成“source calibration 掩盖了较好的 target ranking”，
  不能继续概括为 alignment 有害。

---

## 5. 统一统计与文件纪律

- 所有点估计先从完整 participant × seed 数组直接计算；bootstrap 均值不替代观测值；
- 10,000 次 participant × seed hierarchical paired bootstrap，固定 RNG seed `20260820`；
- arm 差异保持相同 participant、alignment seed 和 readout seed 配对；
- 报告五个 per-seed 效应，不把 seed 当独立患者扩充样本量；
- matched 为主要 OOD 集，matched-long 为患者不重叠的内部敏感性；不把二者合并；
- 保存 model config、split/fold hash、representation hash、逐患者 logits/probabilities、失败名单；
- 输出先写 predictions，再写 metrics；结果文件默认 no-clobber；
- 既有正式结果逐位复现是新 evaluator 的必要门，不能只复现四舍五入均值。

## 6. 什么不做

本轮明确不做：

- 不重新打开 schema-vs-natural-language、grounding、teacher caption 或 reasoning；
- 不增加第三个 audio backbone 或第二个 projector；
- 不在 matched 上选 epoch、hidden size、dropout、阈值或 calibrator；
- 不把 target cross-fitted calibration 写成部署结果；
- 不因为 Coswara 只差少量配对就放宽 SMD、换标签或换录音类型；
- 不把 CI 覆盖 0 写成“没有信息”或“等效”；
- 不根据 E1--E3 的结果再发明 H14/H15 救场。

## 7. 对 ICASSP 稿件的决策规则

### A. E1--E3 大体重复当前方向

稿件主线保持：**learning participant correspondence is not the same as learning
portable disease evidence**。贡献定位为 pairing-controlled audit protocol；结果为两个
backbone、线性与非线性 readout、direct fusion 和 calibration decomposition 的系统评测。

### B. 非线性 readout 或 OPERA direct fusion 出现明确 matched 增益

不视为“实验失败”，而把论文改写成边界条件：raw／非线性表征含有迁移信息，当前线性
Stage-1 alignment 或 source calibration 没有稳定提取它。禁止继续使用普遍 null 叙述。

### C. 结果复杂且 backbone／seed 不一致

只写成 case study；如果四页无法诚实容纳边界条件，优先 workshop／ML4H/ICBINB 风格场合，
而不是删掉不利结果强投主会。

### D. 无论结果如何

外部确认仍是最大限制。E1--E3 可以提高内部技术完整性，但不能把单数据集 discovery audit
变成跨队列临床结论。
