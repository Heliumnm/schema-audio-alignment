# ICASSP Route A 最终执行状态

日期：2026-09-01

状态：**冻结设计中的 Route-A audit 已全部执行；新 mitigation 方法尚未执行。**

## 一句话结论

在 UKCOVID 上，正确的音频--metadata 配对确实让模型学会了患者 profile correspondence，
尤其保留了性别以及部分年龄／采集信息；但是，这种 correspondence 没有稳定转化成跨人群、
跨 backbone、跨 readout 的 COVID 疾病收益。源域校准得到的额外信心在 matched 人群中缺少
排序收益支撑，因此主要表现为 calibration regret。

这篇稿件的准确主张是：

> **Successful correspondence learning does not imply transferable disease evidence.**

不能扩写为“metadata alignment 普遍有害”“声音中没有疾病信息”或“已经证明一般机制”。

## 1. 研究问题与三个配对控制

Clinical metadata 同时包含疾病、患者背景、症状和 cohort/protocol 信息。普通对比损失只知道
positive pair 应该接近，不知道其中哪部分能跨人群迁移。因此训练三个只改变配对的 arm：

| arm | 保留什么 | 破坏什么 | 用来测什么 |
|---|---|---|---|
| `correct` | 标签共现、逐患者 metadata | 无 | 总对应学习 |
| `within-label` | 相同 COVID 标签的群体统计 | 逐患者对应 | label/population association |
| `global` | metadata 边缘分布 | 标签共现与逐患者对应 | 无对应参照 |

核心差值：

- `C-W = correct - within-label`：逐患者 correspondence；
- `W-G = within-label - global`：标签／人群共现；
- `C-R = correct - raw`：对齐相对冻结原始音频表征的改变。

两个冻结音频编码器均完整执行：AST 前六层和 OPERA-CT。每个 alignment arm 使用五个 seed、
500 epochs，只取最终 checkpoint；matched 结果不用于选择 epoch、loss、head 或 calibrator。

## 2. 当前完成度

| 模块 | 状态 | 结论 |
|---|---|---|
| UKCOVID 队列、split、伪影审计 | 完成 | 招募来源在 Standard train 几乎等于标签，在 matched 中被平衡 |
| AST 三配对训练 | 完成 | 源域对应学会，matched 疾病收益不稳定 |
| OPERA-CT 三配对训练 | 完成 | 重复 correspondence／transfer 分离的主方向 |
| Unique-profile retrieval | 完成 | 两个 backbone 均为 `correct > within > global` |
| Information-channel taxonomy | 完成 | correct 明显保留 sex；COVID `C-W` 不稳定 |
| Metadata/direct/raw-preserving fusion | 完成 | raw audio 在 metadata 上增量不显著；correct 不胜 raw |
| Probability transport | 完成 | NLL 差主要是置信度尺度失配，不是已证实的排序损失 |
| 固定 MLP 非线性读出 | 完成 | 更强 readout 没有救回稳定疾病 transfer |
| 受控 synthetic stress test | 完成但门槛未过 | 只证明 objective 能学 correspondence，不能当机制证明 |
| Coswara 外部确认 | 数据门 NO-GO | 没有运行任何外部模型分数 |
| CODA TB 外部敏感性 | v1 NO-GO；**match-first v2 数据门 GO** | v2 从全部1,081人先冻结100对，SMD 0.0819、level diff 0.040；模型尚未运行 |
| Cambridge Task 2 外部敏感性 | v1 NO-GO；**match-first v2正式数据门GO** | 英国合作者已用真实WAV复核：975人/975条cough全部通过QC，冻结100对平衡target；模型尚未运行 |
| Disease-invariant positive-pair 新方法 | **未执行** | 只作为下一阶段设计，不属于当前结果 |

## 3. Alignment 是否真的学会了 correspondence？

在完整 Standard validation 的 5,179 个 query、1,871 个唯一 metadata profile 上进行检索。
主指标为 macro-profile MRR，避免常见 profile 主导结果。

| backbone | correct | within-label | global | `C-W`，95% CI |
|---|---:|---:|---:|---:|
| AST-6L | 0.008910 | 0.005735 | 0.004188 | +0.003175 [0.001605, 0.004917] |
| OPERA-CT | 0.008771 | 0.005542 | 0.004435 | +0.003229 [0.001545, 0.005000] |

两个 backbone 的五个 seed 均为正。绝对 MRR 很低，因为候选 profile 有 1,871 个且高度重复，
但配对差异明确。因此不能把后续 matched null 解释成“projector 什么都没学到”。

## 4. 学到的是哪类 correspondence？

下面报告 matched 上 `correct - within-label` 的 probe ΔAUROC：

| 信息通道 | AST-6L | OPERA-CT | 判读 |
|---|---:|---:|---|
| sex | +0.1912 [0.1669, 0.2151] | +0.1125 [0.0943, 0.1311] | 两个 backbone 稳定 |
| age 65+ | +0.0464 [0.0062, 0.0848] | −0.0004 [−0.0462, 0.0436] | backbone dependent |
| long recording | +0.0035 [−0.0215, 0.0291] | +0.0235 [0.0008, 0.0471] | 只在 OPERA 明确 |
| loud recording | +0.0043 [−0.0187, 0.0271] | +0.0240 [0.0029, 0.0459] | 只在 OPERA 明确 |
| cough | +0.0098 [−0.0121, 0.0319] | +0.0061 [−0.0109, 0.0233] | 不稳定 |
| recruitment source | +0.0138 [−0.0198, 0.0472] | +0.0138 [−0.0138, 0.0417] | 不稳定 |
| COVID | +0.0062 [−0.0150, 0.0273] | −0.0020 [−0.0196, 0.0158] | 没有跨 backbone 增量 |

`correct - raw` 对几乎所有 probe 都为负。例如 sex 为 AST −0.0590、OPERA −0.0317。
因此准确表述是：alignment **相对 within-label projector 选择性保留了患者对应信息**；不能说
alignment 比 raw audio 编码了更多人口学信息。

## 5. Correspondence 是否转化为 matched disease transfer？

### 5.1 Direct/raw-preserving fusion

matched 上，metadata 记为 `M`、raw audio 为 `R`、correct 为 `C`、within 为 `W`。

| backbone | 比较 | ΔAUROC，95% CI | Δ(−NLL)，95% CI |
|---|---|---:|---:|
| AST | `M+R − M` | +0.0028 [−0.0016, 0.0073] | −0.0034 [−0.0194, 0.0123] |
| OPERA | `M+R − M` | +0.0041 [−0.0006, 0.0087] | +0.0118 [−0.0047, 0.0283] |
| AST | `M+C − M+W` | +0.0156 [0.0034, 0.0276] | −0.2698 [−0.3205, −0.2214] |
| OPERA | `M+C − M+W` | +0.0021 [−0.0068, 0.0107] | −0.1389 [−0.1868, −0.0954] |
| AST | `M+C − M+R` | −0.0040 [−0.0093, 0.0012] | −0.0077 [−0.0314, 0.0151] |
| OPERA | `M+C − M+R` | −0.0008 [−0.0062, 0.0047] | +0.0243 [−0.0004, 0.0501] |

AST 的 `C-W` 有一个小的 ranking signal，但 OPERA 未复现；两者的 NLL 都显著变差，且
correct 没有超过 raw audio。因此不能把 AST 单独的 +0.0156 写成可迁移方法收益。

### 5.2 固定 MLP 非线性 readout

更强的固定 MLP 仍未得到跨 backbone 的疾病迁移：

| backbone | audio `C-W` ΔAUROC | audio `C-R` ΔAUROC | `M+C − M+W` ΔAUROC | 对应 Δ(−NLL) |
|---|---:|---:|---:|---:|
| AST | +0.0050 [−0.0154, 0.0270] | −0.0113 [−0.0291, 0.0068] | +0.0168 [0.0013, 0.0320] | −0.2239 [−0.2701, −0.1784] |
| OPERA | −0.0052 [−0.0208, 0.0110] | −0.0247 [−0.0404, −0.0081] | +0.0019 [−0.0087, 0.0117] | −0.1432 [−0.1901, −0.0956] |

结论不是“线性头太弱”。AST fusion 的小正向排序仍然存在，但没有 backbone robustness，
而且没有胜过 raw audio；OPERA 的 correct audio 反而显著低于 raw。

## 6. Calibration／probability transport 回答了什么？

source calibrator 直接迁移到 matched 时：

- AST audio `C-W`：ΔAUROC +0.0062，Δ(−NLL) −0.0249；
- OPERA audio `C-W`：ΔAUROC −0.0020，Δ(−NLL) −0.0176。

只修正目标患病率并不能解决问题：AST Δ(−NLL) 为 −0.0277，OPERA 为 −0.0209。
AST 的固定 shrink curve 显示，预测越尖锐，负向 NLL 差距越大。

但是，用 participant-disjoint matched-long 拟合一维 target calibrator，再原样应用到 matched：

| backbone | target-transport `C-W` ΔAUROC | target-transport Δ(−NLL) |
|---|---:|---:|
| AST | +0.0062 [−0.0156, 0.0264] | +0.00013 [−0.00227, 0.00243] |
| OPERA | −0.0020 [−0.0194, 0.0164] | −0.00020 [−0.00261, 0.00228] |

这说明 source-calibrated NLL 的伤害主要来自目标域置信度／slope 失配，而不是已经证明的
疾病排序损失。重新校准消除了 NLL 差距，但没有创造 transferable disease ranking。

## 7. Synthetic mechanism 的正式结局

冻结网格为 3 个 confounding strength × 3 个 shortcut strength × 10 seeds，共 90 次模拟。

- Gate 1（correspondence）通过：当 shortcut strength ≥1 时，correct retrieval 高于 within；
- Gate 2（high-confounding failure mode）失败；
- Gate 3（dose trend）失败；
- Gate 4（rho=0 边界／ROPE）失败；
- raw-preserving 边界结果混合；
- Phi-2 文本敏感性单元也未达到正文资格。

因此 synthetic 只能说明 objective 能学 correspondence，不能证明“confounding 越强，正确
alignment 必然越差”这一一般机制。它不得作为正文的正向机制图；可在补充材料中报告失败门。

## 8. 外部确认与论文边界

Coswara 2,746 条录音完成 QC，2,646 条通过；冻结匹配在 100 对下限时最大 SMD 为 0.140，
高于 0.12 门槛，因此 NO-GO。没有生成 Coswara 表征、分类器或模型分数。

CODA TB v1 的 9,772 条 solicited cough 全部成功解码并通过 QC；1,081 名 participant 进入
eligible cohort（TB+ 291、TB− 790）。v1 先划分60/40、再在40% target candidate 中匹配，
最终100对的最大绝对 SMD 为0.276、最大分类水平比例差为0.13，因此 NO-GO。

在没有读取任何 CODA 模型输出的前提下，仓库另行冻结了 secondary match-first v2：匹配字段、
距离、100对、0.12／0.08门槛和唯一删减路径全部不变，只把 matched target 的冻结放到
development split 之前。v2 从全部1,081人得到278个初始配对，确定性删减到100对后，最大
SMD为**0.0819**、最大分类比例差为**0.040**，全部数据门通过且两次运行输出哈希逐位一致。
这允许另行预注册并运行 CODA 模型，但截至本文档更新时，AST／OPERA、projector、retrieval
和TB classifier仍未运行。

当前证据是：

- 一个 UKCOVID discovery audit；
- 两个 frozen audio backbone；
- 线性和固定 MLP 两种 readout；
- retrieval、probe、fusion 和 calibration 的闭环；
- 没有独立外部模型确认；
- Coswara 在预注册平衡门停止；CODA v1 停止，但透明标记的 model-blind match-first v2 已通过
  数据门，模型结果尚不存在；
- synthetic 没有通过机制门。

Cambridge Task 2 的 `structure.json` 已确认983名参与者和1,486次cough采集；通过
`UID + Folder Name` 精确连接并执行英语／严格标签筛选后为975人（500阴性、475阳性）。历史
split-first v1只得到26对，保持NO-GO。英国合作者随后在DTA环境中用真实WAV运行单独版本化的
match-first v2：975条入组cough全部通过QC，从241个初始匹配中冻结恰好100对平衡target
（max SMD与fine-balance difference均为0），剩余人为539 train／114 validation／122
source-test。因此正式数据门为`GO`；Cambridge模型结果仍不存在。

## 9. 当前可以与不能说的话

可以说：

> Correct audio--metadata pairing learned measurable participant correspondence, but this
> did not yield a robust cross-backbone, cross-readout disease-transfer gain under
> covariate-balanced evaluation.

不能说：

- metadata alignment 普遍无效或普遍有害；
- AST／OPERA 没有 COVID 信息；
- probe 可解码性证明分类器因果使用了某属性；
- matched evaluation 消除了所有 confounding；
- synthetic 已证明一般机制；
- Coswara 或 CODA TB 已完成外部模型复现。

## 10. 仓库位置

- 冻结设计：`docs/ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md`
- synthetic 预注册：`docs/SYNTHETIC_CORRESPONDENCE_TRANSFER_PREREG_ZH.md`
- 论文蓝图：`docs/ICASSP_PAPER_BLUEPRINT_ZH.md`
- 英文草稿：`docs/ICASSP_DRAFT_EN.md`
- 最终小型结果与 SHA-256：`results/route_a_final/`
- Coswara 数据门结局：`docs/COSWARA_DATA_GATE_OUTCOME_ZH.md`
- CODA TB 数据门结局：`docs/CODA_TB_DATA_GATE_OUTCOME_ZH.md`
- CODA TB match-first v2：`docs/CODA_TB_MATCH_FIRST_V2_OUTCOME_ZH.md`
- CODA TB v2 公开摘要：`results/coda_tb_match_first_v2_summary.json`
- CODA TB 公开聚合摘要：`results/coda_tb_data_gate_summary.json`
- Cambridge 合作者一键外部验证包：`external_validation/cambridge_covid_sounds/`
- Cambridge Task-2结构与v2预检查：`external_validation/cambridge_covid_sounds/TASK2_STRUCTURE_MATCH_FIRST_PRECHECK_ZH.md`
- Cambridge v2聚合预检查：`results/cambridge_task2_match_first_precheck.json`
- 一次性执行器：`scripts/run_icassp_route_a_once.sh`
- 下一阶段方法设计（未执行）：`docs/DISEASE_INVARIANT_ALIGNMENT_FUTURE_PLAN_ZH.md`
