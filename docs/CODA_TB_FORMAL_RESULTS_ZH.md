# CODA TB 正式模型结果：correspondence 成立，matched TB transfer 不确定

日期：2026-09-03  
协议：`coda-match-first-v2-formal-v1`  
地位：CODA TB Challenge Train release 内部的、预注册的 secondary external sensitivity audit

## 一句话结论

> 在 CODA TB 上，Correct 音频--metadata 配对对 AST-6L 和 OPERA-CT 都显著提高了患者
> profile retrieval，并稳定增强了年龄和性别的可解码性；但是，在冻结的100对协变量平衡
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
- AST-6L 与 OPERA-CT 各自完成 Correct／Within-label／Global × 5 seeds × 500 epochs；
- 30个 epoch-500 正式训练产物全部通过完整性、配对、hash、有限值和调度审计；
- source-test 和 matched-target 只在正式训练、验证集 readout 与 source calibration 冻结后读取。

原始 split-first v1 的 NO-GO 仍然保留；match-first v2 是在只查看 v1 聚合数据门之后另行冻结
的 secondary analysis，不冒充最初 untouched confirmation，也不冒充 CODA hidden validation。

## 2. Alignment 是否真的发生？

在 validation 的122名 participant／122个唯一 profile 上进行 audio-to-profile retrieval：

| backbone | Correct MRR | Within-label MRR | Global MRR | Correct−Within，95% CI |
|---|---:|---:|---:|---:|
| AST-6L | 0.1091 | 0.0423 | 0.0523 | **+0.0667 [0.0349, 0.1021]** |
| OPERA-CT | 0.0735 | 0.0433 | 0.0433 | **+0.0302 [0.0064, 0.0555]** |

两个 backbone 的5个 seed 的 Correct−Within 都为正，且分层配对 bootstrap CI 下界均大于0。
因此 correspondence 门明确通过：后续 transfer 结果不能解释成“projector 什么也没学到”。

Within−Global 在两个 backbone 上都没有明确优势，所以主要可识别增量来自逐 participant 的
正确配对，而不是一个稳定的 label-only 检索效应。

## 3. 协变量平衡后的 TB 迁移

预注册主比较是 matched-target 上 `Correct − Within-label`。`Δ(−NLL)>0` 表示 Correct 的
NLL 更低、更好。

| backbone | ΔAUROC，95% CI | Δ(−NLL)，95% CI | 冻结判定 |
|---|---:|---:|---|
| AST-6L | +0.0341 [−0.0221, 0.0916] | +0.0198 [−0.0227, 0.0631] | inconclusive |
| OPERA-CT | −0.0256 [−0.0872, 0.0346] | −0.0070 [−0.0596, 0.0406] | inconclusive |

两个共同主指标都要求 CI 下界大于0；两个 backbone 均未通过。CI 也没有同时完整落入预注册
等价区间（AUROC ±0.02、`−NLL` ±0.01），因此不能把结果改写成“已证明等价/零效应”。AST
点估计向好，OPERA 点估计向坏，准确结论只能是 **matched disease transfer 不确定且不具
backbone robustness**。

matched-target 绝对结果：

| backbone | Raw AUROC | Correct AUROC | Within AUROC | Global AUROC |
|---|---:|---:|---:|---:|
| AST-6L | 0.5300 | 0.5378 | 0.5037 | 0.4999 |
| OPERA-CT | 0.5530 | 0.5448 | 0.5704 | 0.5373 |

Correct 相对 Raw 同样没有明确优势：AST ΔAUROC +0.0078 [−0.0524, 0.0683]；OPERA
−0.0082 [−0.0737, 0.0563]。metadata fusion 中的 Correct−Within 也未通过：AST +0.0251
[−0.0275, 0.0805]；OPERA −0.0256 [−0.0839, 0.0316]。

## 4. 为什么 source-test 看起来更好？

source-test 上的 Correct−Within AUROC：

| backbone | ΔAUROC，95% CI | Δ(−NLL)，95% CI |
|---|---:|---:|
| AST-6L | +0.0756 [−0.0102, 0.1610] | +0.0225 [−0.0205, 0.0632] |
| OPERA-CT | **+0.1050 [0.0170, 0.1866]** | +0.0202 [−0.0246, 0.0622] |

两个 backbone 的5个 seed 在 source-test 的 AUROC 方向都为正，OPERA 的 AUROC CI 排除0；
但 NLL 未通过，而且进入 matched-target 后方向不一致。这一形状与论文的核心审计问题一致：
源分布中的 apparent gain 不能替代协变量平衡后的 transfer 证据。

## 5. Correct pairing 主要保留了什么？

matched-target 上 `Correct − Within-label` 的稳定信息通道：

| probe | AST-6L ΔAUROC，95% CI | OPERA-CT ΔAUROC，95% CI |
|---|---:|---:|
| sex | **+0.1058 [0.0582, 0.1536]** | **+0.0446 [0.0058, 0.0852]** |
| age ≥45 | **+0.1116 [0.0612, 0.1634]** | **+0.0671 [0.0142, 0.1260]** |

两种 backbone 都稳定增强性别与年龄的可解码性，而 TB 的 Correct−Within 不稳定。其他临床、
country 和 acquisition probes 多数依赖 backbone 或 CI 跨0。准确表述是：Correct pairing
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
- `results/coda_tb_formal_results_opera_ct.json`。

对应 SHA-256：

| 文件 | SHA-256 |
|---|---|
| AST results | `7227f782ebe4ecea7091f2eacbdbf9e8b2acca959232bd110b7107be5577bedb` |
| OPERA results | `ebdca4bc20b68a01f58b7bf3e11a95c5916ea4529910ce0b11e4af16c7f21e1c` |
| training audit | `207e680ca66decdf786c58d910b5927996dd2f94c89c2a442fe4a827b9af186d` |

患者级 metadata、音频、participant/pair manifest、逐患者 embedding/prediction、checkpoint 和
pairing 文件继续留在受控数据目录，不进入 GitHub。
