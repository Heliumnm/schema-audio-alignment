# CODA TB 正式模型结果：correspondence 成立，matched TB transfer 不确定

日期：2026-09-03  
协议：`coda-match-first-v2-formal-v1`  
地位：CODA TB Challenge Train release 内部的、预注册的 secondary external sensitivity audit

## 一句话结论

> 在 CODA TB 上，Correct 音频--metadata 配对对 AST-6L、OPERA-CT 和 HeAR 都显著提高了患者
> profile retrieval；三个 backbone 都增强了性别可解码性，AST／OPERA 还增强了年龄可解码性；但是，在冻结的100对协变量平衡
> TB+/TB− target 上，疾病迁移没有得到跨 backbone 一致且置信区间排除0的收益。

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
- AST-6L、OPERA-CT 与 HeAR 各自完成 Correct／Within-label／Global × 5 seeds × 500 epochs；
- 45个 epoch-500 正式训练产物全部通过完整性、配对、hash、有限值和调度审计；
- source-test 和 matched-target 只在正式训练、验证集 readout 与 source calibration 冻结后读取。

原始 split-first v1 的 NO-GO 仍然保留；match-first v2 是在只查看 v1 聚合数据门之后另行冻结
的 secondary analysis，不冒充最初 untouched confirmation，也不冒充 CODA hidden validation。

## 2. Alignment 是否真的发生？

在 validation 的122名 participant／122个唯一 profile 上进行 audio-to-profile retrieval：

| backbone | Correct MRR | Within-label MRR | Global MRR | Correct−Within，95% CI |
|---|---:|---:|---:|---:|
| AST-6L | 0.1091 | 0.0423 | 0.0523 | **+0.0667 [0.0349, 0.1021]** |
| OPERA-CT | 0.0735 | 0.0433 | 0.0433 | **+0.0302 [0.0064, 0.0555]** |
| HeAR | 0.1134 | 0.0489 | 0.0367 | **+0.0646 [0.0320, 0.1014]** |

三个 backbone 的5个 seed 的 Correct−Within 都为正，且分层配对 bootstrap CI 下界均大于0。
因此 correspondence 门明确通过：后续 transfer 结果不能解释成“projector 什么也没学到”。

Within−Global 在三个 backbone 上都没有明确优势，所以主要可识别增量来自逐 participant 的
正确配对，而不是一个稳定的 label-only 检索效应。

## 3. 协变量平衡后的 TB 迁移

预注册主比较是 matched-target 上 `Correct − Within-label`。`Δ(−NLL)>0` 表示 Correct 的
NLL 更低、更好。

| backbone | ΔAUROC，95% CI | Δ(−NLL)，95% CI | 冻结判定 |
|---|---:|---:|---|
| AST-6L | +0.0341 [−0.0221, 0.0916] | +0.0198 [−0.0227, 0.0631] | inconclusive |
| OPERA-CT | −0.0256 [−0.0872, 0.0346] | −0.0070 [−0.0596, 0.0406] | inconclusive |
| HeAR | −0.0127 [−0.0916, 0.0631] | −0.0278 [−0.0816, 0.0206] | inconclusive |

两个共同主指标都要求 CI 下界大于0；三个 backbone 均未通过。CI 也没有同时完整落入预注册
等价区间（AUROC ±0.02、`−NLL` ±0.01），因此不能把结果改写成“已证明等价/零效应”。AST
点估计向好，OPERA 与 HeAR 点估计向坏，准确结论只能是 **matched disease transfer 不确定且不具
backbone robustness**。

matched-target 绝对结果：

| backbone | Raw AUROC | Correct AUROC | Within AUROC | Global AUROC |
|---|---:|---:|---:|---:|
| AST-6L | 0.5300 | 0.5378 | 0.5037 | 0.4999 |
| OPERA-CT | 0.5530 | 0.5448 | 0.5704 | 0.5373 |
| HeAR | 0.5760 | 0.5210 | 0.5337 | 0.5376 |

Correct 相对 Raw 同样没有明确优势：AST ΔAUROC +0.0078 [−0.0524, 0.0683]；OPERA
−0.0082 [−0.0737, 0.0563]；HeAR −0.0550 [−0.1188, 0.0082]。metadata fusion 中的
Correct−Within 也未通过。

## 4. 为什么 source-test 看起来更好？

source-test 绝对 AUROC（括号内为95% CI）：

| backbone | Raw | Correct | Within | Global |
|---|---:|---:|---:|---:|
| AST-6L | 0.6463 [0.5447, 0.7453] | 0.6383 [0.5360, 0.7383] | 0.5628 [0.4728, 0.6512] | 0.5602 [0.4727, 0.6505] |
| OPERA-CT | 0.6409 [0.5434, 0.7346] | 0.6681 [0.5713, 0.7615] | 0.5631 [0.4739, 0.6507] | 0.5983 [0.5060, 0.6870] |
| HeAR | 0.6580 [0.5616, 0.7497] | 0.6492 [0.5567, 0.7380] | 0.5989 [0.5045, 0.6861] | 0.6027 [0.5145, 0.6868] |

source-test 上的 Correct−Within AUROC：

| backbone | ΔAUROC，95% CI | Δ(−NLL)，95% CI |
|---|---:|---:|
| AST-6L | +0.0756 [−0.0102, 0.1610] | +0.0225 [−0.0205, 0.0632] |
| OPERA-CT | **+0.1050 [0.0170, 0.1866]** | +0.0202 [−0.0246, 0.0622] |
| HeAR | +0.0503 [−0.0367, 0.1380] | +0.0168 [−0.0343, 0.0650] |

三个 backbone 的 source-test AUROC 点估计都为正；只有 OPERA 的 AUROC CI 排除0；
但 NLL 未通过，而且进入 matched-target 后方向不一致。这一形状与论文的核心审计问题一致：
源分布中的 apparent gain 不能替代协变量平衡后的 transfer 证据。

相对冻结 Raw，source-test 上同样没有明确优势：AST 的 Correct−Raw ΔAUROC 为 −0.0080
[−0.0917, 0.0737]，OPERA-CT 为 +0.0272 [−0.0603, 0.1183]。因此 source-test 的主要
可见差异是 Correct 相对打乱配对，而不是 Correct 稳定超过未经对齐的音频表示。

## 5. Correct pairing 主要保留了什么？

matched-target 上 `Correct − Within-label` 的稳定信息通道：

| probe | AST-6L ΔAUROC，95% CI | OPERA-CT ΔAUROC，95% CI |
|---|---:|---:|
| sex | **+0.1058 [0.0582, 0.1536]** | **+0.0446 [0.0058, 0.0852]** |
| age ≥45 | **+0.1116 [0.0612, 0.1634]** | **+0.0671 [0.0142, 0.1260]** |

HeAR 的 sex Correct−Within 同样显著：**+0.0364 [0.0149, 0.0603]**；age≥45 为
+0.0386 [−0.0119, 0.0901]，方向为正但区间跨0。因此性别通道跨三个 backbone 重复，年龄
通道只在 AST 与 OPERA 上得到明确支持。

AST 与 OPERA 都稳定增强性别与年龄的可解码性，HeAR 只明确增强性别；TB 的
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

- `results/coda_tb_formal_training_audit.json`；
- `results/coda_tb_formal_results_ast.json`；
- `results/coda_tb_formal_results_opera_ct.json`；
- `results/coda_tb_formal_results_hear.json`。

对应 SHA-256：

| 文件 | SHA-256 |
|---|---|
| AST results | `7227f782ebe4ecea7091f2eacbdbf9e8b2acca959232bd110b7107be5577bedb` |
| OPERA results | `ebdca4bc20b68a01f58b7bf3e11a95c5916ea4529910ce0b11e4af16c7f21e1c` |
| HeAR results | `dc9fba855cf8937bde3c294ef2e741b56119eabbbacf2249d1981f547df3c6ad` |
| training audit | `207e680ca66decdf786c58d910b5927996dd2f94c89c2a442fe4a827b9af186d` |

患者级 metadata、音频、participant/pair manifest、逐患者 embedding/prediction、checkpoint 和
pairing 文件继续留在受控数据目录，不进入 GitHub。
