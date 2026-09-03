# Cambridge Task-2 reconstructed cohort：match-first v2 数据门

> **2026-09-03 superseded notice：** 后续身份审计发现本版本把 Web metadata 中恒定的
> `Uid=form-app-users`误当成一个 participant；官方 Task-2 loader 实际以 Web `Folder Name`
> 为subject key。本文保留为当时冻结的历史记录，但由此产生的975人数据门与target均已作废。
> 修正依据见 `CAMBRIDGE_TASK2_IDENTITY_AUDIT_ZH.md`；任何正式模型不得使用旧config。

冻结日期：2026-09-03
定位：model-blind secondary external sensitivity analysis

## 1. 已知边界

受限数据发布缺少 README 所称的官方 Task-2 split CSV，因此不能复现官方成员划分。
`structure.json` 证明实际 Task-2 音频包含1,486次 cough 采集、983名参与者；按
`UID + Folder Name` 与三平台 metadata 精确连接后，975名参与者具有英语、严格近期阳性或
从未阳性的标签，且Task-2实际收录的采集之间没有标签冲突。

v1 对这975人先做70/10/20 participant split，再只在199人的 test 中匹配，最多只有97个
阳性且最终得到26对，因此在100对门槛上结构性 NO-GO。v2 只检验“先匹配、后划分”能否建立
可用的平衡 endpoint。v1永久保留，v2不冒充最初 untouched confirmation。

## 2. 不改变的规则

- participant 是分析单位；primary audio 仅为 Task-2 cough；
- `UID + Folder Name` session linkage、英语筛选、严格标签、冲突排除、label-blind index
  session hash、波形QC和PCM去重均与v1相同；
- exact matching：age band、sex、cough、fever、sore throat、shortness of breath、asthma、
  other respiratory condition；
- pair cost：smoker mismatch权重1，platform mismatch权重2；
- Hungarian maximum-cardinality assignment、`cambridge-match-v1` tie-break和唯一greedy
  trimming path不变；
- matched target固定为恰好100对；
- 最大绝对SMD `<=0.12`，最大fine-balance difference `<=0.08`；
- 不加载任何encoder，不读取embedding、prediction、AUROC、NLL或calibration。

## 3. v2顺序

1. 对全部audio-eligible participant执行maximum-cardinality matching；
2. 沿冻结greedy path删减到恰好100对；
3. 200名participant冻结为`matched_target`；
4. 剩余participant按`label × platform`分层，以
   `sha256("cambridge-match-first-v2-development|UID")`固定排序，划分70% train、15%
   validation、15% source-test；
5. matched target不选择模型、checkpoint、projector、classifier、regularisation或
   calibrator。

## 4. GO条件

- matched target恰好100对且全部exact fields pairwise一致；
- max |SMD| `<=0.12`，max fine-balance difference `<=0.08`；
- train至少500人且每类至少100人；
- validation与source-test各至少75人、每类至少20人；
- participant与decoded PCM跨split重叠为0；
- train至少100个unique metadata profile，最大profile不超过train的10%；
- 数据门没有加载或读取任何模型产物。

全部通过才允许另行冻结模型实验。任一失败即关闭Cambridge v2，不换hash、不改字段、不降阈值。

## 5. 解释

即使GO，它也只是Task-2 reconstructed cohort上的外部敏感性endpoint，不是完整200GB+ Cambridge
corpus、不是官方Task-2 split复现，也不是模型正结果。

## 6. 冻结时已经知道的信息

这个v2并非原始confirmatory protocol，而是在v1数据门NO-GO后公开版本化的secondary protocol。
冻结正式WAV运行前已经完成结构+metadata预检查，并知道：严格队列为975人，maximum-cardinality
可得277对，固定删减到100对时平衡门可通过。正式WAV数据门仍可能因解码、时长、空波形或PCM
重复而失败；该预检查不能替代正式运行。此顺序必须在论文中公开，不能把v2包装成未看数据的
预注册确认。
