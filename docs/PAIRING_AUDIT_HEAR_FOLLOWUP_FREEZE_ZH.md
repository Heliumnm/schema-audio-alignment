# UKCOVID HeAR E1--E3 事后扩展冻结说明

日期：2026-09-12  
证据地位：**post-hoc third-backbone diagnostic extension；不是独立确认实验**

## 1. 为什么补 HeAR

UKCOVID 主分析预先确定的两个 backbone 是 AST-6L 与 OPERA-CT。HeAR 后来按独立冻结协议完成
Correct／Within-label／Global 三臂训练，并被纳入第三骨干稳健性分析。为避免主结果包含 HeAR、而
E1--E3 机制诊断只覆盖前两个 backbone，本扩展把同一套诊断原样应用到 HeAR。

这不会把 HeAR 提升为 primary backbone，也不会把 E1--E3 变成 confirmatory evidence。

## 2. 时间与结果盲法

- AST／OPERA 的 E1--E3 结果已经产生，因此本扩展必然是事后分析；
- HeAR 原始 C/W/G 结果也已经产生；
- HeAR E1 条件检索在本冻结说明写入前作为执行可行性步骤完成，只作描述性诊断；
- 本说明写入时，HeAR 的 `W_{y,s}` projector、E2 readout 和 E3 target-assisted readout 尚未训练；
- 下列协议冻结后，不根据 HeAR 结果改变 arm、seed、epoch、readout、候选库、bootstrap 或报告规则。

## 3. 冻结协议

完全继承 `docs/PAIRING_AUDIT_SUPPLEMENTARY_EXPERIMENTS_PREREG_ZH.md`：

1. E1：Standard validation 条件检索；候选条件为 unrestricted、same sex、same label、same label
   and sex；主指标为 macro-profile MRR；
2. E2：新增 `W_{y,s}`，即同疾病标签且同记录性别的无自配对置换；五个 seed 0--4、500 epochs；
3. 每个 seed 复用原 HeAR Within-label 的初始化 hash 与 batch-order hash；
4. E2 报告 `C-W_{y,s}` 与 `W_{y,s}-W` 的 retrieval、probe 和 matched disease 差值；
5. E3：只在 matched-long 上训练、选择和校准固定 logistic readout，再原样应用到 participant-
   disjoint matched；
6. E3 依次报告 `C-W`、`C-W_{y,s}`、`W_{y,s}-W` 与 `C-Raw`；
7. participant × seed 分层配对 bootstrap 固定为 10,000 次；
8. E3 永远标为 target-assisted diagnostic，不并入 source-only 主结果。

## 4. 冻结输入

- HeAR revision：`9b2eb2853c426676255cc6ac5804b7f1fe8e563f`；
- 原始表示：72,458 × 512，正式审计见 `docs/HEAR_UKCOVID_EXTENSION_OUTCOME_ZH.md`；
- alignment：Correct／Within-label／Global × seeds 0--4 × 500 epochs；
- 训练集：20,714 participants；
- Standard validation：5,179；matched：1,814；matched-long：4,196；
- metadata text 与 Phi-2 缓存沿用主分析冻结版本；
- E2 配对层计数必须精确为：
  - negative/female 7,378；negative/male 5,866；
  - positive/female 4,690；positive/male 2,775；positive/missing 5。

## 5. 解释边界

- `W_{y,s}-W` 明确为正且 `C-W_{y,s}` 接近零：原 C-W sex 结果主要依赖记录性别匹配关系；
- `C-W_{y,s}` 仍明确为正：记录性别不足以解释全部精确配对差异；
- target-assisted `C-W` 为正但 `C-Raw` 不正：不能声称 alignment 改善疾病表示；
- target-assisted 仍无稳定优势：只能说固定线性诊断未找到隐藏收益，不能证明表示完全不含疾病信息；
- 不跨 backbone 挑选有利结果，不进行 pooled causal inference。

