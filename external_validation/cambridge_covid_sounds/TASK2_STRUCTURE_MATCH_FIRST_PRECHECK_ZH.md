# Cambridge Task 2 文件结构与 match-first v2 预检查

> **2026-09-03 correction：** 最初把根目录中的`metadata`误算为participant，同时把
> `form-app-users`误算为一个participant。正确展开为300个Android UID、682个iOS UID和
> 18个Web Folder Name，共1,000个Task-2 subject。完整更正见
> `CAMBRIDGE_TASK2_IDENTITY_AUDIT_ZH.md`。本文以下975人预检查已superseded。

日期：2026-09-03
性质：仅使用目录结构和 metadata 的模型盲态预检查；不是正式波形数据门，也不是模型结果。

## 结构确认

受限发布中的 `structure.json` 完整列出了 Task 2 子集：

- 983 名参与者；
- 1,486 个含 cough 的采集 session；
- 1,486 个 cough 文件；
- 232 名参与者有多个 session；
- 每名参与者有 1–18 个 session。

因此 Task 2 是从完整 Cambridge 数据中筛出的约一千人子集，不等于完整 200GB+ 母库。
它也不是“975 个文件”：975 是执行英语和严格 COVID 标签规则后的可分析参与者数。

## 与 metadata 的精确连接

以 `UID + Folder Name` 连接三平台 metadata：

- 1,482 / 1,486 个 session 有精确 metadata 行；
- 所有 983 名参与者至少有一个可连接 session；
- 英语且标签为近14天阳性或从未阳性的严格筛选后，剩余 975 人；
- 500 阴性、475 阳性；
- 精确连接的 session 内标签冲突为 0；
- 同一参与者在进入分析的 Task-2 session 间标签冲突为 0。

这解决了 UID-only 审计无法确定采集时间的问题。

## Match-first v2 预检查

在不读取波形、embedding、prediction、AUROC 或 NLL 的条件下，使用冻结 matching 字段运行：

- maximum-cardinality 初始匹配：277 对；
- 沿冻结 greedy objective 删除到恰好 100 对；
- 最终最大绝对 SMD：0.000；
- 最终最大类别比例差：0.000；
- exact fields mismatch：0。

冻结 target 后，剩余 775 人按 `label × platform` 和固定哈希划分：

| Population | N | 阴性 | 阳性 |
|---|---:|---:|---:|
| train | 540 | 279 | 261 |
| validation | 114 | 59 | 55 |
| source-test | 121 | 62 | 59 |
| matched target | 200 | 100 | 100 |

train 有 281 个 unique schema profile，最大 profile 占 3.15%。结构与 metadata 层面的
门槛全部通过。

## 当前判定

**PROVISIONAL GO。** 原因是本机只有目录清单和 metadata，没有 Task-2 WAV，尚未在 v2 中重新执行
decode、时长、finite/nonzero、PCM duplicate 等波形 QC。英国合作者必须用实际 Task-2 音频运行：

```bash
bash run_reconstructed_match_first_v2.sh \
  /DATA/covid19/metadata \
  /DATA/test2 \
  /DATA/cambridge_match_first_v2
```

只有返回的 `public/data_gate.json` 为 `GO`，且 `public/overlap_report.json` 的正式训练许可为
`true`，才允许人工审阅并另行解锁模型阶段。Task 1未下载时，跨任务重叠明确记为未测量；
它不进入训练，因此不阻塞Task-2内部的split/hash完整性校验。

## 解释边界

v1 的 split-first NO-GO 永久保留。v2 是显式版本化的 secondary sensitivity analysis：它先定义
平衡 target，再划分剩余开发人群。即使 v2 正式 GO，也不能声称复现缺失的官方 Task-2 split，
不能称为 untouched confirmatory endpoint，更不能在数据门阶段作任何模型结论。
