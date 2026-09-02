# CODA TB match-first v2 数据门结果

日期：2026-09-03  
决定：**GO，可以进入另行预注册的 CODA 模型实验**

## 一句话结论

CODA v1 的 NO-GO 主要来自“先把60%患者分给 development、再只在剩余40%里匹配”造成的
候选池限制。v2 在不改变匹配字段、距离、100对功效下限、0.12 SMD 或0.08分类比例差门槛的
前提下，先从全部1,081名 eligible participant 中冻结100对 matched target，再划分剩余
development。结果最大绝对 SMD 为0.0819、最大分类水平比例差为0.040，全部数据门通过。

这只是 **model-blind feasibility GO**：说明能够建立平衡的 CODA endpoint，不代表 alignment
有效、TB 可以由咳嗽诊断，也不代表已完成外部模型复现。

## 1. 为什么 v1 与 v2 得到不同结论

| 协议 | matching candidate | 初始对数 | 最终对数 | max \|SMD\| | max level diff | 结论 |
|---|---:|---:|---:|---:|---:|---|
| v1：split-first | 约445人（冻结40%） | 117 | 100 | 0.2756 | 0.130 | NO-GO |
| v2：match-first | 全部1,081人 | 278 | 100 | **0.0819** | **0.040** | **GO** |

v2 没有换随机种子寻找有利划分。它沿 v1 相同的 country × sex exact matching、pair cost、
`coda-match-v1` tie-break 和唯一 greedy trimming path，从278对确定性删减178对，冻结恰好
100对。变化只有 target selection 与 development split 的先后顺序。

由于 v2 是在查看 v1 聚合数据门后建立的，它被定义为 secondary external sensitivity
analysis，不覆盖或撤回 v1，也不称为最初 untouched confirmation。

## 2. 数据与冻结 split

| 项目 | 数量 |
|---|---:|
| clinical participant | 1,105 |
| solicited cough | 9,772 |
| QC 通过 cough | 9,772 |
| eligible participant | 1,081 |
| TB+ / TB− | 291 / 790 |
| matched target | 200（100 / 100） |
| train | 603（127 / 476） |
| validation | 122（24 / 98） |
| source-test | 156（40 / 116） |

所有 participant split overlap 为0，decoded PCM cross-split overlap 为0。matched target 从未
参与模型、epoch、projector、classifier、regularisation 或 calibration 的选择。

## 3. 平衡结果

- country 与 sex：pairwise exact，全部差值为0；
- 最大绝对 SMD：0.08192597，低于0.12；
- 最大分类水平比例差：0.04000000，低于0.08；
- 最大 SMD 来自 `prior TB extrapulmonary`，绝对值0.08193；
- age、height、weight、heart rate、temperature 和 log cough duration 均在冻结门内；
- fever、night sweats、hemoptysis、weight loss、smoking、HIV 和 prior-TB fields 均在冻结门内。

## 4. 复现与受控文件

同一代码从原始受控数据独立运行两次：decision、全部 gates、split counts、matching report，
以及三个受控 CSV 的 SHA-256 逐位一致。

| 受控输出 | SHA-256 |
|---|---|
| audio QC | `b2df1c98bdcbbf5302d12a0f34b873b87e03e7cf3db3e5d4b5b9f16180004848` |
| participant manifest | `f3a423d6dc2f09c1ec4a21af7ff45774cf7cb25fe3d84d3ed0285461e7781a6c` |
| matched pairs | `f2587ffe9ac14102f73dc18c1b65186c23d38f5953be83e3fa5ac6836c43cea4` |

这些 CSV 留在受控 Seagate 数据目录，不进入 GitHub。公开仓库只保存无 participant ID 的聚合
摘要 `results/coda_tb_match_first_v2_summary.json`。

## 5. 现在允许与不允许做什么

允许：

1. 另写并冻结 CODA 正式模型预注册；
2. 在 train 训练 Correct／Within-label／Global；
3. validation 只用于冻结的选择和 source calibration；
4. 最后一次性读取 source-test 与 matched target；
5. 报告 correspondence、信息通道和 TB transfer 是否分离。

仍然不允许：

- 把本次 GO 写成模型正结果；
- 根据 matched target 调模型或选择 checkpoint；
- 把 CODA training-release 内部 target 冒充官方 challenge validation；
- 隐藏 v1 NO-GO；
- 在看到模型结果后修改这100对患者。

