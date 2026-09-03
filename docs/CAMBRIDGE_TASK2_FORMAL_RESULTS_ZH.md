# Cambridge COVID-19 Sounds Task-2 外部敏感性结果

日期：2026-09-03

## 结论

Cambridge Task-2 cough 子集在修正 participant namespace 后包含 989 名可用参与者。模型盲的
match-first 流程先冻结 100 对协变量完全平衡的 target，再把剩余参与者固定划为 train、
validation 和 source-test。AST-6L、OPERA-CT 与 HeAR 随后使用完全相同的
Correct／Within-label／Global、五个 seed 和 500 epochs 协议。

结果跨三个音频 backbone 重复了同一个分离：**Correct pairing 能恢复患者 metadata
correspondence，并稳定增强性别可解码性，但没有带来 matched COVID transfer。**

本结果属于从官方 Task-2 音频子集和全量 metadata 重建的、预先冻结的外部敏感性分析，
不是官方 benchmark split 的复现，也不是独立于设计过程的 untouched confirmation。

## 数据与完整性

| population | N | 说明 |
|---|---:|---|
| train | 551 | 只用于 projector 与下游 readout |
| validation | 117 | 只用于冻结 readout／calibration |
| source-test | 121 | 原始 cohort structure 下的测试 |
| matched target | 200 | 100 个正负匹配对 |

- 服务器重新扫描得到 4,458 条 WAV，与 Mac 端文件数和总字节数逐位一致；
- 三份 metadata CSV 的 SHA-256 在传输前后完全一致；
- 数据门在服务器独立复现为 `989 participants / 100 matched pairs / GO`；
- train、validation、source-test、matched target 的 participant overlap 与 audio-hash overlap
  均为 0；
- matched target 的最大 SMD 和最大 fine-balance difference 均为 0。

## 1. Alignment 是否建立了逐患者 correspondence？

在 117 名 validation participant、96 个唯一 metadata profile 上报告 macro-profile MRR。
主比较是 `Correct − Within-label`；Within-label 保留 COVID 标签组内分布，但破坏逐患者配对。

| backbone | Correct MRR | Within MRR | Global MRR | Correct−Within，95% CI |
|---|---:|---:|---:|---:|
| AST-6L | 0.0963 | 0.0530 | 0.0579 | **+0.0433 [0.0151, 0.0799]** |
| OPERA-CT | 0.0869 | 0.0571 | 0.0492 | +0.0298 [−0.0011, 0.0638] |
| HeAR | 0.0921 | 0.0464 | 0.0489 | **+0.0457 [0.0203, 0.0763]** |

AST 与 HeAR 的区间排除 0；OPERA 方向相同但区间略跨 0。三者的点估计共同说明正确配对
确实给 projector 提供了逐患者 correspondence，而不是“alignment 什么也没有学到”。

## 2. Correspondence 是否转化为 matched COVID transfer？

| backbone | matched AUROC：Correct | Within | Raw | Correct−Within ΔAUROC，95% CI |
|---|---:|---:|---:|---:|
| AST-6L | 0.5325 | 0.5773 | 0.5931 | −0.0448 [−0.1181, 0.0300] |
| OPERA-CT | 0.5166 | 0.5088 | 0.4971 | +0.0077 [−0.0869, 0.1132] |
| HeAR | 0.4719 | 0.5232 | 0.5235 | −0.0512 [−0.1186, 0.0161] |

三个 backbone 的主比较均未排除 0，方向也不一致。Correct 相对 raw 的 ΔAUROC 为 AST
−0.0606、OPERA +0.0195、HeAR −0.0516，区间也都覆盖 0。因此不能主张正确 metadata
alignment 改善了 covariate-balanced COVID 识别。

NLL 结论同样没有正向证据。Correct−Within 的 Δ(−NLL) 为 AST −0.0074、OPERA −0.0139、
HeAR −0.0323；前两者区间包含 0，HeAR 的区间上界约为 0。HeAR 的 Δ(−Brier) 为
−0.0151 [−0.0293, −0.0002]，说明这条 arm 的概率质量反而更差。

## 3. Correct pairing 更稳定地保留了什么？

matched target 上的性别 probe 给出了跨 backbone 一致的 `Correct − Within-label`：

| backbone | 性别 ΔAUROC，95% CI |
|---|---:|
| AST-6L | **+0.1318 [0.0745, 0.1922]** |
| OPERA-CT | **+0.1233 [0.0716, 0.1740]** |
| HeAR | **+0.1282 [0.0893, 0.1703]** |

五个 seed 在三个 backbone 中全部为正。相比之下，cough 和 asthma probe 的区间较宽且方向
不稳定；age≥70 与 Web platform 因 source train／validation 中类别过少而按冻结规则标记为
`not_estimable_in_source`。

准确表述是：**正确配对相对同标签打乱稳定保留了性别相关的患者对应信息。** Probe 只说明
信息可解码，不能证明 COVID 分类器因果使用了性别。

## 4. Direct fusion 的次要读数

matched target 上，raw audio + metadata 相对 metadata-only 的 ΔAUROC 为：AST +0.0923
[0.0194, 0.1707]、OPERA −0.0089 [−0.0894, 0.0706]、HeAR +0.0350
[−0.0371, 0.1093]。只有 AST 的排序增量区间排除 0，且对应 Δ(−NLL) 区间仍包含 0，
因此不是跨-backbone 的 direct-fusion 收益。

## 5. 可以与不能说的话

可以说：

> Across AST, OPERA-CT, and HeAR, correct audio--metadata pairing preferentially preserved
> participant sex and improved profile correspondence, but did not produce a robust gain in
> covariate-balanced COVID transfer.

不能说：

- Cambridge 官方 benchmark 已被复现；
- metadata alignment 普遍有害或必然无效；
- matched target 消除了所有未测量混淆；
- 100 对样本已经证明效应严格等于 0；
- probe 可解码性证明下游分类器使用了该属性。

## 6. 可审计文件

- `results/cambridge_task2_formal_results_ast.json`
- `results/cambridge_task2_formal_results_opera_ct.json`
- `results/cambridge_task2_formal_results_hear.json`
- 数据门与 overlap 的本地公开包保存在受控数据目录，不向仓库提交音频、逐患者 metadata、
  prediction、embedding 或 private manifest。
