# CODA TB 正式模型结果：correspondence 成立，matched TB transfer 不确定

日期：2026-09-15
协议：`locked_panel_v3_20260914`
地位：CODA TB Challenge Train release 内部的、预注册的 secondary external sensitivity audit

## 一句话结论

> 在 CODA TB 上，Correct 音频--metadata 配对对 AST-6L、OPERA-CT 和 HeAR 都显著提高了患者
> profile retrieval；三个 backbone 的 Correct−Within 性别可解码性均为正；但是，在冻结的
> 100 对协变量平衡 TB+/TB− target 上，预先锁定的 Correct−`W_{y,s}` 疾病比较没有得到
> 跨 backbone 一致且置信区间排除 0 的收益。

预注册解释分支因此是：

`correspondence_gain_but_matched_transfer_inconclusive`

它支持“成功的 correspondence learning 不等于已经得到 transferable disease evidence”这一
评估歧义，但不能写成“metadata alignment 已被证明无效／有害”，也不能写成“咳嗽中没有 TB
信号”。

## 1. 数据与执行完整性

- 原始 solicited cough：9,772 条，全部通过客观 QC；
- 正式 eligible cohort：1,081 人、9,749 条 solicited cough；
- train：603 人；validation：122 人；source-test：156 人；
- matched-target：200 人，即100对 TB+/TB−；
- matched-target 最大绝对 SMD：0.0819（门槛0.12）；
- 最大分类水平比例差：0.040（门槛0.08）；
- AST-6L、OPERA-CT 与 HeAR 各自完成 Correct／Within-label／`W_{y,s}`／Global × 5 seeds × 500 epochs；
- 60 个 epoch-500 正式训练端点全部通过完整性、配对、hash、有限值和调度审计；
- source-test 和 matched-target 只在正式训练、验证集 readout 与 source calibration 冻结后读取。

原始 split-first v1 的 NO-GO 仍然保留；match-first v2 是在只查看 v1 聚合数据门之后另行冻结
的 secondary analysis，不冒充最初 untouched confirmation，也不冒充 CODA hidden validation。

## 2. Alignment 是否真的发生？

在 validation 的122名 participant／122个唯一 profile 上进行 audio-to-profile retrieval：

| backbone | Correct MRR | Within MRR | `W_{y,s}` MRR | Correct−Within，95% CI | Correct−`W_{y,s}`，95% CI |
|---|---:|---:|---:|---:|---:|
| AST-6L | 0.1056 | 0.0474 | 0.0690 | **+0.0582 [0.0288, 0.0922]** | **+0.0366 [0.0021, 0.0725]** |
| OPERA-CT | 0.0984 | 0.0464 | 0.0675 | **+0.0520 [0.0222, 0.0840]** | +0.0309 [−0.0022, 0.0662] |
| HeAR | 0.0905 | 0.0505 | 0.0671 | **+0.0400 [0.0129, 0.0687]** | +0.0234 [−0.0052, 0.0528] |

三个 backbone 的5个 seed 的 Correct−Within 都为正，且分层配对 bootstrap CI 下界均大于0。
因此 correspondence 门明确通过：后续 transfer 结果不能解释成“projector 什么也没学到”。

Within−Global 在三个 backbone 上都没有明确优势，所以主要可识别增量来自逐 participant 的
正确配对，而不是一个稳定的 label-only 检索效应。

## 3. 协变量平衡后的 TB 迁移

锁定主比较是 matched-target 上 `Correct − W_{y,s}`。`Δ(−NLL)>0` 表示 Correct 的
NLL 更低、更好；`W_{y,s}−Within` 同时报告为配对控制诊断。

| backbone | `C-W_{y,s}` ΔAUROC，95% CI | `W_{y,s}-W` ΔAUROC，95% CI | `C-W_{y,s}` Δ(−NLL)，95% CI | 冻结判定 |
|---|---:|---:|---:|---|
| AST-6L | +0.0476 [−0.0127, 0.1151] | −0.0010 [−0.0620, 0.0595] | +0.0092 [−0.0378, 0.0563] | inconclusive |
| OPERA-CT | −0.0077 [−0.0600, 0.0423] | −0.0030 [−0.0602, 0.0542] | −0.0054 [−0.0376, 0.0238] | inconclusive |
| HeAR | −0.0102 [−0.0707, 0.0481] | +0.0245 [−0.0304, 0.0786] | −0.0230 [−0.0841, 0.0405] | inconclusive |

两个共同主指标都要求 CI 下界大于0；三个 backbone 均未通过。CI 也没有同时完整落入预注册
等价区间（AUROC ±0.02、`−NLL` ±0.01），因此不能把结果改写成“已证明等价/零效应”。AST
点估计向好，OPERA 与 HeAR 点估计向坏，准确结论只能是 **matched disease transfer 不确定且不具
backbone robustness**。

matched-target 绝对结果：

| backbone | Raw AUROC | Correct AUROC | Within AUROC | `W_{y,s}` AUROC |
|---|---:|---:|---:|---:|
| AST-6L | 0.5300 | 0.5330 | 0.4865 | 0.4854 |
| OPERA-CT | 0.5530 | 0.5368 | 0.5475 | 0.5445 |
| HeAR | 0.5763 | 0.5362 | 0.5220 | 0.5464 |

Correct 相对 Raw 同样没有明确优势：AST ΔAUROC +0.0030 [−0.0561, 0.0642]；OPERA
−0.0162 [−0.0710, 0.0386]；HeAR −0.0401 [−0.0994, 0.0199]。metadata fusion 中的
Correct−Within 也未通过。

## 4. 为什么 source-test 看起来更好？

source-test 绝对 AUROC（括号内为95% CI）：

| backbone | Raw | Correct | Within | Global |
|---|---:|---:|---:|---:|
| AST-6L | 0.6463 [0.5447, 0.7453] | 0.6458 [0.5404, 0.7467] | 0.5475 [0.4513, 0.6406] | 0.5626 [0.4790, 0.6442] |
| OPERA-CT | 0.6409 [0.5434, 0.7346] | 0.6343 [0.5352, 0.7318] | 0.5637 [0.4805, 0.6449] | 0.5531 [0.4648, 0.6399] |
| HeAR | 0.6578 [0.5613, 0.7495] | 0.6468 [0.5521, 0.7377] | 0.6083 [0.5231, 0.6911] | 0.5863 [0.4969, 0.6726] |

source-test 上的 Correct−Within AUROC：

| backbone | ΔAUROC，95% CI | Δ(−NLL)，95% CI |
|---|---:|---:|
| AST-6L | +0.0983 [−0.0024, 0.2025] | +0.0437 [−0.0075, 0.1014] |
| OPERA-CT | +0.0706 [−0.0016, 0.1447] | +0.0151 [−0.0133, 0.0437] |
| HeAR | +0.0385 [−0.0566, 0.1293] | +0.0253 [−0.0310, 0.0838] |

三个 backbone 的 source-test AUROC 点估计都为正，但三个区间都跨 0；NLL 也未通过，
而且进入 matched-target 后方向不一致。这一形状与论文的核心审计问题一致：
源分布中的 apparent gain 不能替代协变量平衡后的 transfer 证据。

相对冻结 Raw，source-test 上同样没有明确优势：AST、OPERA-CT、HeAR 的 Correct−Raw
ΔAUROC 分别为 −0.0005 [−0.0847, 0.0828]、−0.0067 [−0.0939, 0.0840]、
−0.0109 [−0.0967, 0.0790]。因此 source-test 的主要
可见差异是 Correct 相对打乱配对的方向性估计，而不是 Correct 稳定超过未经对齐的音频表示。

## 5. Correct pairing 主要保留了什么？

matched-target 上 `Correct − Within-label` 的稳定信息通道：

| probe | AST-6L ΔAUROC，95% CI | OPERA-CT ΔAUROC，95% CI |
|---|---:|---:|
| sex | **+0.1287 [0.0843, 0.1750]** | **+0.0449 [0.0090, 0.0814]** |
| age ≥45 | **+0.1089 [0.0584, 0.1657]** | +0.0397 [−0.0178, 0.0956] |

HeAR 的 sex Correct−Within 同样显著：**+0.0426 [0.0197, 0.0689]**；age≥45 为
−0.0049 [−0.0539, 0.0430]。因此性别通道跨三个 backbone 重复，年龄通道仅在 AST 上
得到明确支持。

三个 backbone 都增强性别可解码性，只有 AST 明确增强年龄可解码性；TB 的
Correct−Within 在三个 backbone 上均不稳定。其他临床、country 和 acquisition probes 多数
依赖 backbone 或 CI 跨0。准确表述是：Correct pairing
相对 Within-label projector 保留了更多 patient correspondence；probe 只证明信息可解码，
不证明 TB classifier 因果使用了这些属性，也不证明 Correct 比冻结 Raw 编码了更多人口学信息。

## 6. 对论文的作用

UKCOVID 可能被质疑为单一英国 COVID cohort 的特殊现象。CODA TB 改变了疾病、国家、采集
环境、标签参考标准和音频 backbone，却重复得到下面的逻辑链：

```text
Correct pairing
    -> participant/profile correspondence 明确增加
    -> patient demographic information 明确增加
    -> source-domain ranking 可能增加
    -> covariate-balanced disease transfer 不稳定
```

因此论文可以把贡献写成一个跨数据集可执行的 **Pairing-Controlled Transfer Audit**，并报告
在 COVID 与 TB 上均观察到 correspondence 与 transfer 的分离。不能写成普遍因果规律；
CODA matched transfer 的统计结论是 inconclusive，且 match-first v2 是透明标记的 secondary
analysis。

## 7. 公开文件与隐私边界

公开、aggregate-only 文件：

- `results/coda_tb_formal_results_ast.json`；
- `results/coda_tb_formal_results_opera_ct.json`；
- `results/coda_tb_formal_results_hear.json`；
- `results/locked_external_panel_v3/coda_tb/` 下的非识别 alignment manifests、
  text-axis audit 与 formal-training audit。

对应 SHA-256：

| 文件 | SHA-256 |
|---|---|
| AST results | `360d221d61324c6fa445ceab832c1b845389fabb2b05ea7f78ff2cb218b503e3` |
| OPERA results | `8ef3d13cd088682eecd90df6a55da3285b608b4c4df4eb3de9a77ebf42b87a8e` |
| HeAR results | `3f51942117339a126ffe06670a2b68c466c945d09331a4b6a92703ed66c73408` |
| v3 training audit | `82fbcccc2a522e578b520155cf0d7aa186f3f55d71d64b148bb26ed9090128dd` |

患者级 metadata、音频、private participant/pair manifest、逐患者 embedding/prediction、
checkpoint 和 pairing 数组继续留在受控数据目录，不进入 GitHub；仓库只同步不含参与者标识的
aggregate results 与训练哈希 manifest。
