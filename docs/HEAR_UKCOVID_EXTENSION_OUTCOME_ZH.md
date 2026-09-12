# UKCOVID HeAR 第三骨干稳健性分析结果

日期：2026-09-12  
证据地位：**post-hoc third-backbone robustness analysis；不是新的 confirmatory 实验**

## 1. 为什么此前没有进入论文表格

HeAR 的 UKCOVID 正式运行已经在 2026-09-04 完成，但当时只有服务器归档和两个尚未加入 Git
的聚合 JSON。提交清单因此把它标为“未经最终审计，不计入正式结果”。这不是没有运行，而是
结果整合没有完成。本文件完成最后审计并保留其事后证据地位；AST-6L 与 OPERA-CT 仍是
UKCOVID 的两个 primary backbones。

## 2. 完整性审计

- 冻结 HeAR revision：`9b2eb2853c426676255cc6ac5804b7f1fe8e563f`；
- checkpoint SHA-256：`2317d8c67d6457140c45cd9c8e37495ecc58d0a07124aef1a626f1ce11325c13`；
- 预检：100 人重复提取逐位一致，`failures=[]`；
- 正式表示：`72,458 x 512`，72,458 人全部有限、非零且顺序与冻结 cohort 一致；
- embedding SHA-256：`da174d9830136a92105c7552f1fa3e15430d60f2005caaff595c571d2f98d701`；
- 训练：Correct／Within-label／Global × seeds 0--4 × 500 epochs，共 15 个最终 projector；
- 每个 seed 的三臂共享初始化与 batch-order hash，仅配对 hash 不同；
- alignment manifest SHA-256：`dbaed7771665bf770a3f56a0963557a9651935d78b7ade5a8c71d1b4032700d3`；
- participant-level predictions SHA-256：`cd4d8e45f3161b9197600b6138779932984163093d542ef402bd5a6e4922b8d2`；
- Standard validation 为 5,179 人，matched 为 1,814 人，matched-long 为 4,196 人；
- profile retrieval 未读取 matched/test labels；大型表示、checkpoint、排名和逐参与者预测仅保留
  在个人服务器，不进入公开仓库。

## 3. Profile correspondence

Standard validation 包含 5,179 个 query 和 1,871 个唯一 metadata profile。

| 比较 | macro-profile MRR 差值 | 95% CI |
|---|---:|---:|
| Correct − Within-label | **+0.007113** | **[+0.004763, +0.009707]** |

五个 seed 的差值全部为正。因此 HeAR projector 确实学习了正确配对，而不是训练完全失效。

## 4. Matched information 与疾病预测

matched 集共有 1,814 人。

| 指标 | Correct − Within-label | 95% CI |
|---|---:|---:|
| sex probe ΔAUROC | **+0.0862** | **[+0.0734, +0.0995]** |
| COVID ΔAUROC | +0.0029 | [−0.0115, +0.0165] |
| COVID Δ(−NLL) | −0.0150 | [−0.0327, +0.0025] |

疾病 AUROC 的绝对值为：

| Raw HeAR | Correct | Within-label | Global |
|---:|---:|---:|---:|
| 0.5417 | 0.5455 | 0.5426 | 0.5131 |

HeAR 因而重复了 AST／OPERA 的核心形状：正确配对显著增强 profile correspondence 与性别
可解码性，但没有建立 matched COVID disease-transfer 增益。Correct 相对 Raw 的 AUROC 差值
同样很小（约 +0.0038，区间跨 0）。

## 5. 对论文计数的影响

加入这项事后 UKCOVID HeAR 稳健性分析后，全文描述性计数更新为：

- profile retrieval：11/12 个 dataset--backbone setting 的配对区间排除 0；
- sex probe：12/12 个 setting 的配对区间排除 0；
- matched disease：0/12 个 setting 建立正向增益；
- matched disease 点估计为六正六负，不进行跨数据集 pooled inference。

这增加了 backbone 稳健性，但不能把 UKCOVID 从 discovery audit 变成独立确认，也不能把
“区间覆盖 0”写成等效或真实效应为零。

## 6. 公开聚合文件

- `results/ukcovid_hear_disease_results.json`，SHA-256
  `9188a5f415267bc2ff8631161a2e4e82e4b97b40eb3a362e8aeeba7b3e8e8682`；
- `results/ukcovid_hear_profile_retrieval.json`，SHA-256
  `446cee851da0da7a921a04165ed52dcf60abcd4245c495adc95cab292d3a47d1`。

两个文件只含聚合指标、per-seed 指标和审计字段，不含 participant identifier 或逐参与者预测。

## 7. E1--E3 追加机制诊断

2026-09-12 又按单独冻结的 post-hoc 协议完成了 HeAR 的 E1--E3：

- 新的 sex-preserving shuffle `W_{y,s}` 吸收了原来的性别差异：
  `C-W_{y,s}` sex ΔAUROC +0.0064 [−0.0011, 0.0136]，而
  `W_{y,s}-W` +0.0797 [0.0667, 0.0928]；
- 但 retrieval 仍留有 residual exact-pairing signal：`C-W_{y,s}` macro-profile MRR
  +0.00394 [0.00175, 0.00637]；
- matched COVID 的 `C-W_{y,s}` 为 +0.0130 [−0.0023, 0.0287]，未建立正向收益；
- target-assisted readout 的 `C-W`、`C-W_{y,s}` 与 `C-Raw` 区间也全部覆盖 0。

这把原来的“sex correspondence”进一步定位为**记录性别一致性**，同时说明仍有少量非性别
profile correspondence；但两者都没有形成可靠的 COVID transfer。三个 backbone 的联合结果与
完整审计见 `docs/PAIRING_AUDIT_E1_E2_E3_OUTCOME_ZH.md`。
