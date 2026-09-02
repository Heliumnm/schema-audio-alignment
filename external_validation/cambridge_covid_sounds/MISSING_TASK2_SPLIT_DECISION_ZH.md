# Cambridge Task-2 划分文件缺失后的冻结决定

> 历史v1记录：该split-first路径已因不足100对而NO-GO。当前secondary路线见
> `PREREGISTRATION_MATCH_FIRST_V2_ZH.md`；不得覆盖或删除本文件。

日期：2026-09-02

## 观察到的发布内容

获得授权的数据目录包含 `all_metadata/`、`covid-19data/`、`task1/`、`task2/` 和数据字典，
但 `task1/`、`task2/` 内只有音频，没有公开代码说明中引用的
`data_0426_en_task2.csv`。因此无法恢复官方Task-2的精确成员、标签表和fold。

官方论文说明 benchmark 使用患者互斥、人口学平衡的70%/10%/20% train/validation/test，
Task 2共有1,000位参与者、1,486个样本。这个描述不足以唯一恢复原始成员关系；任何自行生成的
划分都必须称为重建cohort，不能称为官方split复现。

## 为什么不使用“随意70/15/15”

1. 随意随机种子不可复现，也容易在看结果后重新抽取；
2. 官方报告的比例是70/10/20，不是70/15/15；
3. 若只有约1,000人，15% test最多形成75对阳性/阴性，和冻结的至少100对门槛数学冲突；
4. 标签随采集时间变化，只按患者ID贴标签会把不同时期的声音混在一起。

## 模型盲metadata审计

在三份原始metadata中观察到53,449行。只保留英语并使用严格标签定义后：

- 严格可解释患者：3,493；
- 同时出现阳性和阴性状态：48，整位患者排除；
- 仅近14天阳性：521；
- 仅从未阳性的阴性：2,924。

这些数字在任何representation、prediction或模型score生成前得到，只用于冻结数据处理，不能
用于选择模型或修改门槛。20% test理论上只有约104位阳性，经过音频QC和严格matching后仍可能
低于100对；若如此，结论是数据门 `NO_GO`。

## 冻结重建规则

1. 用 `Uid + Folder Name` 连接同一次采集的metadata和cough；
2. `Language=en`；
3. `positiveLast14/last14`映射为阳性；
4. `negativeNever`映射为阴性；
5. 其他COVID状态全部排除；
6. 同一患者出现两种严格标签时整人排除；
7. 多个可用同标签session时，由固定、标签盲SHA-256选择一个index session；
8. 患者级按 `label × platform` 分层，以固定SHA-256排序切成70/10/20；
9. test内继续执行100对、max |SMD| 0.12、fine-balance 0.08数据门；
10. 输出必须写明 `official_split_reproduced: false`。

## 一键入口

```bash
bash run_reconstructed_gate.sh \
  /DATA/covid19/metadata \
  /DATA/covid19 \
  /DATA/cambridge_audit_output
```

若以后找到原始 `data_0426_en_task2.csv`，另行使用 `run_task2_gate.sh`。两条路线分别报告，
不得合并成员或选择其中结果更好的一条。
