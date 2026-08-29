# CODA TB 外部数据门结果

日期：2026-08-29  
决定：**NO-GO，不启动 CODA TB 模型实验**

## 一句话结论

CODA TB 的下载、患者映射和 9,772 条 solicited cough 音频质检全部成功，但冻结的
participant-level matching 在保留 100 对 TB+/TB− 患者后仍未达到预注册的协变量平衡标准：
最大绝对 SMD 为 0.276，高于 0.12 门槛；最大分类水平比例差为 0.13，高于 0.08 门槛。
因此外部 disease-transfer confirmation 按计划关闭，没有生成音频表征、训练 projector 或
读取任何 TB 模型分数。

这是一项**数据可行性 NO-GO**，不是模型负结果，也不说明 CODA TB 没有可听 TB 信号。

## 1. 模型盲审计范围

本次只允许读取：

- participant ID 与 TB microbiologic reference label；
- country、人口学与临床 metadata；
- solicited-cough 文件映射；
- 解码、时长、波形有限性、格式和重复文件等客观音频 QC；
- 预注册 participant split 与 matching balance。

明确没有运行：

- AST 或 OPERA 表征提取；
- `correct`／`within-label`／`global` projector；
- profile retrieval、probe、TB AUROC、NLL 或 calibration；
- 任何根据模型结果修改 cohort 的步骤。

## 2. 数据完整性与音频 QC

| 项目 | 结果 |
|---|---:|
| clinical metadata | 1,105 人 |
| additional metadata | 1,105 人 |
| solicited-cough 映射 | 9,772 条 |
| 映射到音频的 participant | 1,082 人 |
| 缺失／额外 WAV | 0 / 0 |
| 成功解码并通过 QC | 9,772 / 9,772 |
| 音频格式 | 44.1 kHz、单声道、16 bit |
| 跨 participant 重复组 | 0 |
| participant 内重复组 | 0 |

所以 NO-GO 不是下载失败、坏音频或关联表错误造成的。

## 3. 可用 cohort 与冻结 split

最终 eligible cohort 为 1,081 人，其中 TB+ 291 人、TB− 790 人。排除 23 名没有有效
solicited cough 的 participant，以及 1 名 required metadata 无法用于冻结规则的 participant。

| split | participant 数 |
|---|---:|
| train | 431 |
| validation | 86 |
| source test | 119 |
| matched target | 200（100 对） |
| target unmatched | 245 |

所有 split 的 participant overlap 为 0。

## 4. Matching 为什么没有过门

匹配先得到 117 对，再按运行前固定的确定性规则移除 17 对，保留预注册下限 100 对，并满足
country × sex 的精确平衡。但连续变量和症状仍缺少足够 common support：

| 诊断量 | 结果 | 冻结门槛 |
|---|---:|---:|
| matched pairs | 100 | ≥100 |
| 最大绝对 SMD | **0.2756** | ≤0.12 |
| 最大分类水平比例差 | **0.1300** | ≤0.08 |

主要残余不平衡来自年龄、心率、体重、咳嗽持续时间、体重下降、HIV status、夜间盗汗、
咯血与发热。样本数刚好达到下限并不等于达到了可比性；平衡门优先于模型实验。

## 5. 正确解释与下一步

本结果只支持：

> 在冻结的 60/40 participant split、country × sex 精确匹配和 100 对下限下，CODA TB
> training data 没有为本次 confirmatory Pairing-Controlled Transfer Audit 提供足够的
> covariate common support。

不能说：

- metadata alignment 在 CODA TB 上失败；
- CODA TB 咳嗽不含 TB 信息；
- 换一个 matcher 或放宽 0.12／0.08 后得到的结果仍属于本次确认实验。

因此当前 ICASSP 稿仍是 UKCOVID 单数据集 discovery audit。下一条预先规划的外部路线是让
有授权的英国合作者在 Cambridge COVID-19 Sounds 上执行已冻结的一键审计包；CODA TB 数据
可保留给未来另行预注册的研究，但不得用 post-hoc matching 救回本轮 confirmation。

## 6. 可审计记录

公开仓库只保存不含 participant-level 内容的聚合摘要：
`results/coda_tb_data_gate_summary.json`。

受控服务器目录中保存 participant manifest、matched pairs、audio QC 与完整日志；它们受
Synapse 数据使用条款约束，不提交、不转发。公开摘要记录了三个输入文件和三个受控输出文件
的 SHA-256，可用于核对本次运行。
