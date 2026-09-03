# Cambridge Task-2 reconstructed cohort：match-first v2.1 身份修正

冻结日期：2026-09-03  
状态：在修正后的真实WAV数据门运行前冻结；Cambridge模型从未启动  
性质：对v2 participant namespace bug的单变量修正，不是看模型结果后的新分析

本地 macOS/exFAT 副本可能额外包含 `._*.wav` AppleDouble 资源叉；它们不是 released 音频
样本，扫描器按文件名固定排除。该运行时兼容修复不改变 WAV 内容、participant、submission、
标签、matching 或阈值。

本地 macOS/exFAT 副本可能额外包含 `._*.wav` AppleDouble 资源叉；它们不是 released 音频
样本，扫描器按文件名固定排除。该运行时兼容修复不改变 WAV 内容、participant、submission、
标签、matching 或阈值。

## 为什么必须建立v2.1？

v2把三平台metadata里的原始`Uid`直接当participant ID。Android和iOS可这样做；Web不行：

- Web metadata的`Uid`对全部5,404行恒为`form-app-users`；
- 官方Task-2 loader把Web `Folder Name`当subject key，并读取
  `form-app-users/<Folder Name>`；
- `structure.json`含18个这样的Task-2 Web folder；
- v2错误地把其中15个符合严格标签规则的Web positive合并成1人。

所以旧v2的975人、100对target、split、manifest hash和`GO`均不可复用。

## 唯一允许改变的变量

participant namespace改为：

```text
Android / iOS: participant = Uid
Web:           participant = Folder Name
```

同时Web音频路径固定为：

```text
<audio_root>/form-app-users/<Folder Name>/<cough wav>
```

除此以外，以下全部沿用旧v2，不得调整：

- English only；
- positive = `last14`／`positiveLast14`；
- negative = `negativeNever`；
- 其他COVID状态排除；
- 多次合格采集由同一个label-blind session hash选择；
- cough-only与相同波形QC／PCM去重；
- exact matching字段、cost字段及权重；
- maximum-cardinality matching与同一greedy trimming规则；
- matched target恰好100对；
- max absolute SMD `<=0.12`；
- max fine-balance difference `<=0.08`；
- target先冻结，其余participant按`label × platform`划分70/15/15；
- AST-6L／OPERA-CT、Phi-2、5 seeds、500 epochs与readout协议。

## 纵向采集与多录音规则

- participant单位使用上述稳定namespace；同一participant最终只保留一行；
- endpoint只接纳`last14`／`positiveLast14`和`negativeNever`，不另行推算检测日期窗口；
- 同一稳定participant若在可连接采集中同时出现两种严格endpoint标签，整名participant排除；
- 同标签的多个合格submission由固定的label-blind `participant + folder` hash选一个；
- 被选submission内若有多条cough，分别编码后作participant-level算术平均；
- 原始COVID状态是否随时间变化、严格标签冲突数和实际排除数均进入聚合overlap报告。

## 正式训练前overlap报告

Task 1只做可选目录inventory，不参与训练、选择或匹配。由于它没有下载，本次跨任务overlap
预注册为`NOT_MEASURED_TASK1_UNAVAILABLE`，不得填0或推断；这不阻塞Task-2正式实验。当前
脚本在真实WAV数据门后生成`public/overlap_report.json`，至少报告：

1. Task-2 unique UID和unique submission数量；
2. 每UID recording数量与Android／iOS／Web人数；
3. Task-1可用时报告Task-1∩Task-2的UID和`UID/submission-folder`交集，否则报告未测量；
4. train／validation／matched-target的逐对UID交集；
5. decoded PCM及raw-file hash的跨split重复；
6. 稳定UID内的原始状态变化与严格endpoint标签冲突。

Task-1／Task-2可以有描述性重叠，因为Task 1完全不进入拟合；但Task-2必须完整展开为1,000个
UID和1,486个submission，开发／target UID交集与跨split音频hash必须均为0。报告不包含任何
UID、路径或逐人信息。

## 运行前已经知道的信息

只用目录和metadata、未读取模型输出时已经知道：

- 官方Task-2结构：1,000 subjects、1,486三模态sessions；
- 1,482 sessions可精确连接metadata；
- 严格队列：989 subjects，500 negative／489 positive；
- 其中Web严格合格：15 positive；
- Cambridge与UKCOVID participant ID overlap：0。

这些数字只用于身份正确性检查，不允许据此修改matching或模型规则。

## GO／NO-GO

合作者必须从当前代码重新生成participant manifest与config，并对真实WAV重新运行完整数据门。
只有以下条件全部满足才可恢复正式模型：

1. `reconstruction_report.format_version == cambridge-reconstruction-v2`；
2. `identity_namespace_version == cambridge-task2-official-loader-v1`；
3. 真实WAV QC／duplicate checks通过；
4. matched target恰好100对；
5. 所有旧v2阈值继续通过；
6. `overlap_report.formal_training_permitted_by_overlap_audit == true`；
7. 新aggregate输出经项目负责人人工复核。

任一条件失败即`NO_GO`；不回退到旧975人cohort，也不删除或覆盖旧结果。

## 解释边界

即使v2.1通过，它仍是官方Task-2音频子集上的**custom reconstructed strict-COVID cough
endpoint**，不是缺失官方Task-2 split的复现，也不是untouched confirmation。身份修正只使
分析单位与官方loader一致，不会把secondary sensitivity升级成confirmatory study。
