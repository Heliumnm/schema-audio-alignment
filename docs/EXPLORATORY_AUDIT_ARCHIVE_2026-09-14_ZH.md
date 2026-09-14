# 2026-09-14 探索性审计归档说明

本文件把截至 2026-09-14 的当前仓库状态标记为 **exploratory / data-informed audit archive**。

## 归档点

- 归档 commit：`4795eff` (`Align ICASSP paper with matched-trajectory framing`)
- 建议 tag：`archive/exploratory-audit-20260914`
- 当前归档版本包含：
  - UKCOVID primary pairing audit；
  - UKCOVID E1/E2/E3 定向诊断；
  - HeAR 作为后续第三 backbone 扩展；
  - CODA TB match-first secondary external sensitivity；
  - Cambridge Task-2 reconstructed external sensitivity；
  - Coswara post-hoc global-gate external stress test；
  - 当前 ICASSP paper draft 和 aggregate-only 结果文件。

## 科学地位

这版结果可以支持一个清楚的探索性结论：

> Correct audio--metadata pairing repeatedly improves profile correspondence and patient/demographic decodability, but existing matched disease-transfer estimates do not provide comparably stable evidence of transferable disease improvement.

但这版结果 **不能** 被重新包装成完全 confirmatory external evidence，原因是：

1. 不同数据集和 backbone 的分析状态并不一致；
2. `W_{y,s}`、HeAR、部分 target-assisted readout 和部分 external cohorts 是在主结果或数据门结果之后新增的诊断；
3. CODA、Cambridge、Coswara 的结果已经被读取过，并参与了后续叙事和诊断设计；
4. CI 覆盖 0 不是 equivalence proof；不能写成“已证明无效”或“真实效应为 0”。

因此，本归档版本在论文中应被称为：

- exploratory audit；
- secondary sensitivity analysis；
- post-hoc external stress test；
- hypothesis-generating evidence。

## 为什么需要 clean rerun

如果论文要采用更强、更干净的叙事：

> UKCOVID identified a pairing-controlled hypothesis and fixed the evaluation protocol; CODA TB, Cambridge, and Coswara then tested the locked hypothesis without protocol modification.

那么必须从本归档点之后重新冻结协议，并让外部数据按同一套 pipeline 重跑。旧结果可以作为设计来源和背景，但不能作为独立 confirmatory evidence。

新的 clean rerun 协议见：

- `docs/LOCKED_EXTERNAL_RERUN_PROTOCOL_2026-09-14_ZH.md`

