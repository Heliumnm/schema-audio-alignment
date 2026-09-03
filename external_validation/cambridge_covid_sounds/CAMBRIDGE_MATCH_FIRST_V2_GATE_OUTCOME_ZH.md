# Cambridge Task-2 match-first v2 正式数据门结果

状态：**SUPERSEDED（旧 Web namespace 合并错误；不得启动模型）**
执行日期：2026-09-03  
执行提交：`3626f53638c6cccc511d2f1a4a85a6e5c66f13eb`

> 后续身份审计确认：官方 Task-2 loader 用 Web `Folder Name` 作为subject key，而本次运行
> 把18个 Web submission 的恒定 `Uid=form-app-users`合并成一人。正确严格队列应为989人
> （500阴性、489阳性），不是975人。以下结果仅作为错误被发现前的历史记录；原target、split、
> config和`GO`全部作废。详见 `CAMBRIDGE_TASK2_IDENTITY_AUDIT_ZH.md`。

英国合作者在受限数据环境中，用 Task-2 音频树、三平台完整 metadata 和真实 WAV 执行了
冻结的 match-first v2 数据门。返回包只包含不含 ID、路径和逐样本值的聚合文件。

## 已作废的历史结果

| 项目 | 结果 |
|---|---:|
| 严格英语、近期阳性/从未阳性且音频可连接的 participant | 975 |
| 发现并通过 QC 的 cough WAV | 975 / 975 |
| 初始可匹配对数 | 241 |
| 冻结 matched target | 100 对（200 人，100+/100-） |
| train | 539（260+/279-） |
| validation | 114（55+/59-） |
| source-test | 122（60+/62-） |
| matched target max absolute SMD | 0.0000 |
| matched target max fine-balance difference | 0.0000 |
| train unique metadata profiles | 320 / 539 |
| train modal profile share | 0.0241 |

全部冻结 gate checks 通过。WAV 时长范围为 2.04--21.08 秒，中位数 6.50 秒；没有解码失败、
QC 失败或跨 participant 重复导致的排除。

## 当时的解释（现已作废）

这些数现在只说明旧的 Web-collapsed cohort 当时通过了门；不能再据此运行
`Correct / Within-label / Global`。尚未生成表示、预测、AUROC 或 NLL，因此错误没有污染
Cambridge 模型结果。

该 cohort 仍不是缺失的官方 Task-2 split 的复现，而是事先冻结、模型盲态的 secondary
match-first sensitivity endpoint。因此后续结果不能写成 untouched external confirmation。

## 修正后的下一步

1. 保留旧输出，不覆盖历史；
2. 更新至包含 Web namespace 修复的版本；
3. 以相同标签、matching字段和阈值在新输出目录重跑真实WAV数据门；
4. 人工审阅新 aggregate gate；只有新gate为`GO`才另行解锁正式模型；
5. 不得把旧config复制到新运行。

任何模型、checkpoint、embedding、prediction、matched pair、participant manifest 或原始数据
都不得离开受限环境。
