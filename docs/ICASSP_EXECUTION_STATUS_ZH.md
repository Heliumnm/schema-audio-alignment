# ICASSP Route A 最终执行状态

日期：2026-09-12

> **2026-09-12 补充实验最终状态：E1--E3 全部完成。** AST-6L、OPERA-CT 与事后 HeAR
> 均完成条件检索、标签＋性别保持的 `W_{y,s}` 对照和 target-assisted readout。新对照把三个
> backbone 原有的 Correct−Within sex 差异降到区间覆盖 0；仍有很小、依赖 backbone 的 residual
> profile retrieval，但 source-only 与 target-assisted 两套读出都没有建立稳定 matched COVID
> 收益。完整结果见 `docs/PAIRING_AUDIT_E1_E2_E3_OUTCOME_ZH.md`。

状态：**UKCOVID Route-A、CODA TB secondary external sensitivity、Cambridge Task-2
重建外部敏感性与 Coswara 三骨干 post-hoc stress test 已执行；UKCOVID HeAR 第三骨干事后
稳健性分析已完成并通过最终审计；新 mitigation 方法尚未执行。**

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

两个预先确定的冻结音频编码器均完整执行：AST 前六层和 OPERA-CT。随后按独立冻结协议完成
HeAR 事后第三骨干稳健性分析。每个 alignment arm 使用五个 seed、500 epochs，只取最终
checkpoint；matched 结果不用于选择 epoch、loss、head 或 calibrator。

## 2. 当前完成度

| 模块 | 状态 | 结论 |
|---|---|---|
| UKCOVID 队列、split、伪影审计 | 完成 | 招募来源在 Standard train 几乎等于标签，在 matched 中被平衡 |
| AST 三配对训练 | 完成 | 源域对应学会，matched 疾病收益不稳定 |
| OPERA-CT 三配对训练 | 完成 | 重复 correspondence／transfer 分离的主方向 |
| HeAR UKCOVID 事后第三骨干 | **完成并审计** | correspondence 与 sex probe 明确；matched COVID 增量不明确 |
| Unique-profile retrieval | 完成 | 两个 backbone 均为 `correct > within > global` |
| E1 条件化 profile retrieval | **三骨干完成** | AST／OPERA 同性别后跨零；HeAR 缩小但仍为正 |
| E2 标签＋性别保持的 Within 对照 | **三骨干完成并审计** | sex `C-Wys` 三骨干均跨零；疾病增量均不明确 |
| E3 target-assisted readout | **三骨干完成并审计** | 未发现稳定 `C-W`、`C-Wys` 或 `C-Raw` 疾病优势 |
| Information-channel taxonomy | 完成 | correct 明显保留 sex；COVID `C-W` 不稳定 |
| Metadata/direct/raw-preserving fusion | 完成 | raw audio 在 metadata 上增量不显著；correct 不胜 raw |
| Probability transport | 完成 | NLL 差主要是置信度尺度失配，不是已证实的排序损失 |
| 固定 MLP 非线性读出 | 完成 | 更强 readout 没有救回稳定疾病 transfer |
| 受控 synthetic stress test | 完成但门槛未过 | 只证明 objective 能学 correspondence，不能当机制证明 |
| Coswara 外部确认 | 原贪心门 NO-GO；全局约束二次门 `SECONDARY_GO` | 三骨干 post-hoc stress test 完成；均建立 correspondence，均无 matched COVID transfer 增益 |
| CODA TB 外部敏感性 | **三骨干正式模型已完成** | 三个 backbone 均建立 correspondence；matched TB transfer 不确定且方向不一致 |
| Cambridge Task-2-subset 外部敏感性 | **修正版数据门与三骨干正式模型已完成** | Correct 增强 profile retrieval 与性别信息，但三个 backbone 均无 matched COVID transfer 增益 |
| Disease-invariant positive-pair 新方法 | **未执行** | 只作为下一阶段设计，不属于当前结果 |

## 3. Alignment 是否真的学会了 correspondence？

在完整 Standard validation 的 5,179 个 query、1,871 个唯一 metadata profile 上进行检索。
主指标为 macro-profile MRR，避免常见 profile 主导结果。

| backbone | correct | within-label | global | `C-W`，95% CI |
|---|---:|---:|---:|---:|
| AST-6L | 0.008910 | 0.005735 | 0.004188 | +0.003175 [0.001605, 0.004917] |
| OPERA-CT | 0.008771 | 0.005542 | 0.004435 | +0.003229 [0.001545, 0.005000] |
| HeAR（post-hoc） | 0.012774 | 0.005662 | 0.004535 | +0.007113 [0.004763, 0.009707] |

三个 backbone 的五个 seed 差值均为正；HeAR 为事后稳健性分析，不改变 AST／OPERA 的 primary
地位。绝对 MRR 很低，因为候选 profile 有 1,871 个且高度重复，但配对差异明确。因此不能把
后续 matched null 解释成“projector 什么都没学到”。

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

事后 HeAR 分析给出同样形状：matched sex `C-W` 为 **+0.0862 [0.0734, 0.0995]**，而
COVID 为 **+0.0029 [−0.0115, 0.0165]**。完整审计见
`docs/HEAR_UKCOVID_EXTENSION_OUTCOME_ZH.md`。

### 4.1 性别保持对照把原 sex 效应定位清楚

新增 `W_{y,s}` 在同疾病标签打乱的同时保留记录性别。三个 backbone 的 matched sex
`C-W_{y,s}` 均覆盖 0：AST −0.0004 [−0.0153, 0.0147]、OPERA +0.0067
[−0.0054, 0.0192]、HeAR +0.0064 [−0.0011, 0.0136]；相反，`W_{y,s}-W` 分别为
+0.1916、+0.1058、+0.0797，区间都排除 0。因此原 `C-W` sex 结果主要是 correct pairing
保留了性别一致性。

控制性别后并非所有 correspondence 都消失。`C-W_{y,s}` retrieval MRR 为 AST +0.00164
[0.00001, 0.00335]、OPERA +0.00099 [−0.00096, 0.00300]、HeAR +0.00394
[0.00175, 0.00637]。剩余信号很小且不具三个 backbone 的一致明确性。

对应的 matched COVID `C-W_{y,s}` 为 AST −0.0003、OPERA +0.0009、HeAR +0.0130，
三个区间均覆盖 0。完整表见 `docs/PAIRING_AUDIT_E1_E2_E3_OUTCOME_ZH.md`。

## 5. Correspondence 是否转化为 matched disease transfer？

HeAR 的 matched disease 绝对 AUROC 为 Raw 0.5417、Correct 0.5455、Within 0.5426、Global
0.5131；Correct−Within 为 +0.0029 [−0.0115, 0.0165]，Correct 也没有建立相对 Raw 的优势。
由于它是事后第三骨干分析，未追加 fusion／MLP 选择，不与 AST／OPERA 的 primary robustness
矩阵混为同一层证据。

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

### 5.3 Target-assisted readout 没有救回疾病收益

E3 只在 matched-long 上训练、选择和校准 logistic readout，再应用到 participant-disjoint
matched。它是使用目标域标签的诊断，不能并入 source-only 主结果。其 `C-W` ΔAUROC 为：

- AST −0.0082 [−0.0337, 0.0177]；
- OPERA +0.0287 [−0.0017, 0.0626]；
- HeAR +0.0047 [−0.0278, 0.0408]。

`C-W_{y,s}` 与所有 Δ(−NLL) 区间同样覆盖 0；`C-Raw` 也没有正向证据，AST 反而为
−0.0504 [−0.0843, −0.0182]。因此，源域 readout mismatch 不是当前 null transfer 的充分解释。
这仍不等于证明表示中不存在任何疾病信息。

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

Coswara 2,746 条录音完成 QC，2,646 条通过；原冻结贪心匹配在 100 对下限时最大 SMD 为
0.140，高于 0.12 门槛，因此原数据门 NO-GO。随后在仍未读取模型分数时，单独冻结的全局
约束敏感性求解找到了 100 对可行集合：年龄 SMD 0、最大分类 SMD 0.1034、最大层级比例差
0.07，开发集规模门也通过。三骨干正式压力测试现已完成：profile retrieval 的
Correct−Within MRR 为 AST **+0.0183 [0.0028,0.0365]**、OPERA **+0.0243
[0.0050,0.0452]**、HeAR **+0.0406 [0.0226,0.0619]**；matched COVID ΔAUROC 分别为
+0.0028 [−0.0450,0.0464]、−0.0115 [−0.0750,0.0444]、+0.0037
[−0.0646,0.0724]。三骨干性别 probe 均显著为正。它只是一项 post-hoc external stress test，
不构成 untouched external confirmation。

CODA TB v1 的 9,772 条 solicited cough 全部成功解码并通过 QC；1,081 名 participant 进入
eligible cohort（TB+ 291、TB− 790）。v1 先划分60/40、再在40% target candidate 中匹配，
最终100对的最大绝对 SMD 为0.276、最大分类水平比例差为0.13，因此 NO-GO。

在没有读取任何 CODA 模型输出的前提下，仓库另行冻结了 secondary match-first v2：匹配字段、
距离、100对、0.12／0.08门槛和唯一删减路径全部不变，只把 matched target 的冻结放到
development split 之前。v2 从全部1,081人得到278个初始配对，确定性删减到100对后，最大
SMD为**0.0819**、最大分类比例差为**0.040**，全部数据门通过且两次运行输出哈希逐位一致。
随后在读取任何表示或模型分数前冻结正式协议，并完成 AST-6L、OPERA-CT 与 HeAR 的
Correct／Within-label／Global × 5 seeds × 500 epochs。

CODA profile retrieval 的 Correct−Within MRR 在 AST 为 **+0.0667 [0.0349, 0.1021]**，
OPERA 为 **+0.0302 [0.0064, 0.0555]**，HeAR 为 **+0.0646 [0.0320, 0.1014]**，所以三个 backbone 都明确建立了 participant
correspondence。matched-target 上的 TB Correct−Within 结果为：

| backbone | ΔAUROC，95% CI | Δ(−NLL)，95% CI |
|---|---:|---:|
| AST-6L | +0.0341 [−0.0221, 0.0916] | +0.0198 [−0.0227, 0.0631] |
| OPERA-CT | −0.0256 [−0.0872, 0.0346] | −0.0070 [−0.0596, 0.0406] |
| HeAR | −0.0127 [−0.0916, 0.0631] | −0.0278 [−0.0816, 0.0206] |

三个 backbone 的 sex Correct−Within probe 均显著为正；age≥45 在 AST／OPERA 显著，HeAR
方向为正但区间跨0。matched TB transfer 的共同主指标均未通过，方向也不一致。冻结解释分支是
`correspondence_gain_but_matched_transfer_inconclusive`：不是正向 transfer，也不能因 CI
较宽而写成已证明零效应。完整边界见 `docs/CODA_TB_FORMAL_RESULTS_ZH.md`。

当前证据是：

- 一个 UKCOVID discovery audit；
- 两个 primary frozen audio backbone，加一个事后 HeAR 第三骨干稳健性分析；
- 线性和固定 MLP 两种 readout；
- retrieval、probe、fusion 和 calibration 的闭环；
- 一个完成的三骨干 CODA TB secondary external sensitivity：重复 correspondence／transfer 分离，
  但 matched disease endpoint 为 inconclusive，不是 untouched confirmatory replication；
- Coswara 原贪心门停止；透明标记的全局约束二次门与三骨干压力测试已完成；CODA v1
  停止，其透明标记的 match-first v2 与正式模型历史均完整保留；
- synthetic 没有通过机制门。

Cambridge身份审计现已钉死：它是Cambridge COVID-19 Sounds NeurIPS的官方Task-2音频子集，
不是UKCOVID，也不是Task-1症状任务；当前endpoint因官方Task-2 split CSV缺失而属于custom
strict-COVID reconstruction，并固定为cough-only。正确目录解释为300个Android UID、682个
iOS UID和18个Web Folder Name，共1,000个subject、1,486次三模态采集；与UKCOVID的72,999个
participant identifier精确／忽略大小写重叠均为0。

身份审计同时发现旧重建代码把18个Web submission合并成一个`form-app-users` participant。
正确严格队列为989人（500阴性、489阳性），而不是旧gate的975人。旧v2 gate、split、100对
target和config均已标记superseded。修正后在Mac与服务器独立复现为989人、100对、最大SMD与
fine-balance difference均为0，随后完成AST-6L、OPERA-CT和HeAR正式实验。

Cambridge profile retrieval 的Correct−Within MRR为AST **+0.0433 [0.0151,0.0799]**、
OPERA **+0.0298 [−0.0011,0.0638]**、HeAR **+0.0457 [0.0203,0.0763]**。matched COVID
ΔAUROC则为AST **−0.0448 [−0.1181,0.0300]**、OPERA **+0.0077
[−0.0869,0.1132]**、HeAR **−0.0512 [−0.1186,0.0161]**；三者均无正向迁移证据。
性别probe的Correct−Within为+0.1318、+0.1233、+0.1282，三个区间均排除0。因此Cambridge
跨三个backbone重复了“patient correspondence成立、disease transfer不成立”的主形状；
但它仍是自定义重建split的外部敏感性，不是官方benchmark复现或untouched confirmation。

在 audit 完成后，仓库另行预注册了一个不使用 target score 的 Transfer Repair v2 数据可行性门。
UKCOVID、CODA TB、Cambridge source train 的 joint eligible coverage 分别为 **0.246%**、
**67.50%**、**45.37%**，均未达到冻结的总体及逐标签80%。CODA／Cambridge 的跨环境同标签
positive coverage 都是100%，但同环境、异标签、协变量可比的 negative coverage只有67.50%和
45.37%。因此三套单数据集 Repair v2 均在训练前停止；这是一项 data-support NO_GO，不是方法
效果实验。

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
- CODA TB 已证明 metadata alignment 普遍无效，或已经完成官方 challenge confirmation。

## 10. 仓库位置

- 冻结设计：`docs/ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md`
- synthetic 预注册：`docs/SYNTHETIC_CORRESPONDENCE_TRANSFER_PREREG_ZH.md`
- 论文蓝图：`docs/ICASSP_PAPER_BLUEPRINT_ZH.md`
- 英文草稿：`docs/ICASSP_DRAFT_EN.md`
- 最终小型结果与 SHA-256：`results/route_a_final/`
- Coswara 数据门结局：`docs/COSWARA_DATA_GATE_OUTCOME_ZH.md`
- Coswara 三骨干正式压力测试：`docs/COSWARA_FORMAL_RESULTS_ZH.md`、
  `results/coswara_formal_results_ast.json`、`results/coswara_formal_results_opera_ct.json`、
  `results/coswara_formal_results_hear.json`
- CODA TB 数据门结局：`docs/CODA_TB_DATA_GATE_OUTCOME_ZH.md`
- CODA TB match-first v2：`docs/CODA_TB_MATCH_FIRST_V2_OUTCOME_ZH.md`
- CODA TB 正式结果：`docs/CODA_TB_FORMAL_RESULTS_ZH.md`
- CODA TB v2 公开摘要：`results/coda_tb_match_first_v2_summary.json`
- CODA TB 公开聚合摘要：`results/coda_tb_data_gate_summary.json`
- CODA TB 正式 aggregate JSON：`results/coda_tb_formal_results_ast.json`、
  `results/coda_tb_formal_results_opera_ct.json`、`results/coda_tb_formal_results_hear.json`、
  `results/coda_tb_formal_training_audit.json`
- Cambridge 合作者一键外部验证包：`external_validation/cambridge_covid_sounds/`
- Cambridge Task-2结构与v2预检查：`external_validation/cambridge_covid_sounds/TASK2_STRUCTURE_MATCH_FIRST_PRECHECK_ZH.md`
- Cambridge v2聚合预检查：`results/cambridge_task2_match_first_precheck.json`
- Cambridge Task-2身份核对：`external_validation/cambridge_covid_sounds/CAMBRIDGE_TASK2_IDENTITY_AUDIT_ZH.md`
- Cambridge身份核对机器可读汇总：`results/cambridge_task2_identity_audit.json`
- Cambridge三骨干正式结果：`docs/CAMBRIDGE_TASK2_FORMAL_RESULTS_ZH.md`、
  `results/cambridge_task2_formal_results_ast.json`、
  `results/cambridge_task2_formal_results_opera_ct.json`、
  `results/cambridge_task2_formal_results_hear.json`
- Transfer Repair v2 外部数据门：`docs/TRANSFER_REPAIR_V2_EXTERNAL_GATE_PREREG_ZH.md`、
  `docs/TRANSFER_REPAIR_V2_EXTERNAL_GATE_OUTCOME_ZH.md`、
  `results/transfer_repair_v2_coda_pair_gate.json`、
  `results/transfer_repair_v2_cambridge_pair_gate.json`
- 一次性执行器：`scripts/run_icassp_route_a_once.sh`
- 下一阶段方法设计（未执行）：`docs/DISEASE_INVARIANT_ALIGNMENT_FUTURE_PLAN_ZH.md`
