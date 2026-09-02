# Cambridge Task-2 match-first v2 正式数据门结果

状态：**GO（真实 WAV 已复核；尚无模型结果）**  
执行日期：2026-09-03  
执行提交：`3626f53638c6cccc511d2f1a4a85a6e5c66f13eb`

英国合作者在受限数据环境中，用 Task-2 音频树、三平台完整 metadata 和真实 WAV 执行了
冻结的 match-first v2 数据门。返回包只包含不含 ID、路径和逐样本值的聚合文件。

## 正式结果

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

## 能与不能说明什么

这说明 Task-2 reconstructed cohort 有足够的样本量、profile diversity 和协变量共同支持，
可以运行预注册的 `Correct / Within-label / Global` 外部敏感性实验。它不是模型正结果，尚未
生成表示、预测、AUROC 或 NLL。

该 cohort 仍不是缺失的官方 Task-2 split 的复现，而是事先冻结、模型盲态的 secondary
match-first sensitivity endpoint。因此后续结果不能写成 untouched external confirmation。

## 下一步（不得再改数据定义）

1. 保持当前 participant、100 对 target、split、schema 和 thresholds 原样不动；
2. 在私有 `config.local.json` 中填入 AST、OPERA-CT、Phi-2 的授权本地路径；
3. 将 `protocol.formal_unlock` 设为 `FROZEN_PROTOCOL_AND_DATA_GATE_GO`；
4. 从提交 `3626f53` 运行 `bash run_once.sh config.local.json formal`；
5. 只返回 `public/PUBLIC_RESULTS.zip`、commit SHA、环境摘要和聚合结果可公开性确认。

任何模型、checkpoint、embedding、prediction、matched pair、participant manifest 或原始数据
都不得离开受限环境。
