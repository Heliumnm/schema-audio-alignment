# 2026-09-14 Clean External Rerun 冻结协议

冻结时间：2026-09-14  
协议状态：从本文件提交之后开始生效。  
旧结果状态：`docs/EXPLORATORY_AUDIT_ARCHIVE_2026-09-14_ZH.md` 所列结果只作为探索性归档，不作为本协议的 confirmatory evidence。

## 1. 研究问题

临床音频--metadata alignment 的成功，是否意味着模型学到了能跨人群迁移的疾病证据？

本 clean rerun 把问题拆成三层：

1. **Pairing correspondence**：正确配对是否让 audio representation 更接近对应的 metadata profile？
2. **Information change**：正确配对相对控制配对改变了哪些可解码信息，例如 sex、age、symptoms、source/cohort？
3. **Disease transfer**：在 covariate-balanced target population 上，正确配对是否改善疾病预测？

## 2. Discovery 与 external tests

- **UKCOVID**：discovery / protocol-design dataset。它用于确定本协议、主要对照、主要指标和解释边界。
- **CODA TB**：external test，疾病为 TB，cohort/domain 为 country/site。
- **Cambridge COVID-19 Sounds Task-2 reconstructed cohort**：external test，疾病为 COVID-19，cohort/domain 为 platform 或预冻结 collection domain。
- **Coswara global-gate cohort**：external stress test。原始贪心数据门仍为 NO-GO，因此其地位低于 CODA/Cambridge，但 rerun 后仍按同一协议报告。

所有 external rerun 的模型输出必须写入新的 output root；不得覆盖旧归档结果。

## 3. 固定 backbone panel

三种 frozen audio representation 全部预先列入 rerun，不再根据结果选择：

1. **AST-6L**：primary backbone。它在 UKCOVID 目标结果之外预先选择，作为主要报告 backbone。
2. **OPERA-CT**：respiratory-domain pretrained robustness backbone。
3. **HeAR**：health-acoustic robustness backbone。

主文中不得挑选最好看的 backbone。若篇幅不够，AST-6L 作为 primary；OPERA-CT 和 HeAR 作为同一协议下的 robustness panel。

## 4. 固定 pairing arms

每个 dataset × backbone 均使用相同四个 arms：

| arm | 定义 | 用途 |
|---|---|---|
| Raw | frozen audio representation，无 metadata alignment | 原始音频参照 |
| Correct (C) | `audio_i ↔ metadata_i` | 正确 participant profile 配对 |
| Within-label (W) | `audio_i ↔ metadata_j, y_i=y_j, i≠j` | 保留疾病标签组内 metadata structure，破坏 participant pairing |
| Within-label-sex (W_{y,s}) | `audio_i ↔ metadata_j, y_i=y_j, s_i=s_j, i≠j` | 进一步保留记录 sex 一致性，检查 C-W 是否依赖粗粒度 sex/profile matching |
| Global (G) | `audio_i ↔ metadata_k, i≠k` | 打乱 label-conditioned pairing |

训练规则：

- 不允许自配对回退；
- 不允许跨 sex 回退来填补 `W_{y,s}`；
- 如果某 dataset 中 `W_{y,s}` 无法在原训练集合内为所有 eligible participants 构造，则必须在同一 reduced cohort 上重训 C/W/Wys/G；不得只在新臂删样本；
- 每个 arm 使用相同 frozen audio/text embeddings、projector architecture、initialization seed、audio batch order、dropout stream、optimizer、epoch 数。

## 5. Primary contrast 与 primary metric

### Primary clinical contrast

外部数据的 primary clinical contrast 固定为：

\[
\Delta_{\mathrm{disease}} = \mathrm{AUROC}(C) - \mathrm{AUROC}(W_{y,s})
\]

解释：

- `C-W_{y,s}` 问的是：在保留疾病标签与记录 sex 一致性的打乱配对后，exact participant pairing 是否仍带来 matched disease-transfer gain？
- 它不是严格因果分解；
- 它不证明 sex 被下游 classifier 因果使用；
- 它只是在已有 metadata 条件下更保守地控制一种最强、最稳定的 pairing shortcut。

### Primary clinical metric

- Primary metric：participant-level matched-target AUROC 的 `C-W_{y,s}`。
- Secondary probability metrics：matched-target Δ(−NLL)、Brier、calibration slope。
- Diagnostic endpoints：profile retrieval MRR、sex/age/symptom/source probes。

若 primary AUROC CI 覆盖 0，只能写 “inconclusive / no positive evidence under this protocol”，不能写 “equivalent” 或 “true zero”，除非 CI 完全落入预先等价区间。

## 6. 数据门与 matched target

数据门必须在任何 representation、projector、classifier 或 model score 产生前完成。

每个 external dataset 固定沿用已有数据门：

- CODA TB：match-first v2，100 对 TB+/TB−，max |SMD| ≤ 0.12，max fine-balance difference ≤ 0.08。
- Cambridge：identity-fixed reconstructed Task-2 match-first，100 对 COVID+/COVID−，max |SMD| ≤ 0.12，max fine-balance difference ≤ 0.08。
- Coswara：global-gate cohort；原 greedy gate NO-GO 不改写，global-gate 作为 external stress test 报告。

Matched target 从不用于：

- 选择 projector；
- 选择 epoch；
- 选择 readout hyperparameter；
- 选择 calibration；
- 选择 backbone；
- 修改 matching variables；
- 修改 primary metric。

## 7. Readout 与 calibration

Source-only readout 是正式 clinical endpoint：

1. projector 只在 source train 上训练；
2. logistic readout 的 regularization 和 Platt calibration 只在 source validation 上选择／拟合；
3. source-test 和 matched target 只在 readout 冻结后读取一次；
4. matched target 上报告 AUROC、NLL、Brier、calibration slope。

Target-assisted readout 可以作为 diagnostic appendix：

- 它使用目标分布标签，因此不得进入 source-only primary conclusion；
- 不得用它重新选择 projector、arm、backbone 或 primary metric。

## 8. 随机种子、训练长度与不许改的项

固定：

- seeds：0, 1, 2, 3, 4；
- epochs：500；
- optimizer、batch size、Phi-2 text cache、metadata schema 与 missing-value handling 沿用当前仓库冻结实现；
- bootstrap：使用各 dataset config 中冻结的次数；重采样单位为 participant × seed；
- 只提交 aggregate JSON、配置和 manifest hash；不提交 participant identifiers、metadata rows、predictions、embeddings、checkpoints 或 pairing manifests。

本协议生效后，不允许因为任何 external dataset 的结果而修改：

- `C-W_{y,s}` primary contrast；
- matched target 构造；
- AUROC primary metric；
- readout model class；
- backbone panel；
- bootstrap/reporting rule；
- arm 定义；
- seed 或 epoch。

## 9. 成功、失败和允许的结论

如果 external datasets 重复出现：

```text
profile retrieval gain > 0
sex/profile probe gain > 0
but matched disease C-Wys not consistently > 0
```

允许结论：

> Alignment can reproducibly learn participant/profile correspondence without reproducibly improving covariate-balanced disease transfer.

不允许结论：

- metadata alignment 普遍无效；
- disease effect 已被证明为 0；
- sex causally explains all pairing gains；
- matched target 移除了所有未测量混淆；
- post-hoc 旧结果是 untouched confirmation。

如果 external datasets 给出稳定正向 `C-W_{y,s}` matched disease gain，也必须写明：

> Clean rerun under a stricter pairing-control protocol supports a transfer benefit in that dataset/backbone setting.

不能反向修改 UKCOVID discovery 的历史状态。

## 10. 执行环境修正记录

第一次外部 panel 启动使用了错误的文本缓存环境：Phi-2 缓存由
Transformers 4.49.0 生成，而 UKCOVID discovery 使用的是 4.56.0。该差异改变了
文本 embedding 的哈希，因此该执行在任何新的 external matched endpoint 被读取前
立即终止，归档为 `technical abort`，不构成科学结果，也不得与后续结果合并。

正式执行协议升级为 `locked-external-rerun-20260914-v3`，并将环境职责分离：

- `TEXT_PYTHON`：仅生成 Phi-2 文本缓存，强制 Transformers 4.56.0；
- `ALIGN_PYTHON`：生成 AST/OPERA 表示、训练 projector 与运行 readout；
- `HEAR_PYTHON`：仅生成 HeAR 表示。

主运行器会在启动前 fail closed 检查文本环境版本，并把三个解释器路径、文本
Transformers 版本、代码 commit 和配置哈希写入 run manifest。v3 的 arm、样本门、
primary contrast、metric、seed、epoch 和 readout 均未因 v2 的技术中止而改变。
