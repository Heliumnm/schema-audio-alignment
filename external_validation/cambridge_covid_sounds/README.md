# Cambridge COVID-19 Sounds：一键受限数据外部验证包

> **当前状态（2026-08-29）：等待有 DTA 授权的英国合作者执行。** Coswara 与 CODA TB
> 均在各自冻结的数据平衡门停止，未读取模型分数；因此 Cambridge 是当前唯一仍开放的
> confirmatory external route。合作者必须先只运行 `gate`，不能直接解锁 `formal`。

这个文件夹把 UKCOVID 的 Pairing-Controlled Transfer Audit 移植到 Cambridge
COVID-19 Sounds DTA 数据。它不包含、下载或重新分发任何 Cambridge 数据。

官方信息：

- 数据需要学校／机构签署 Data Transfer Agreement；
- 完整数据含 cough、breathing、voice、人口学、病史与症状；
- 官方 benchmark 文件包含患者级 `train / validation / test` 划分；
- Cambridge 官方说明和申请方式：<https://www.covid-19-sounds.org/en/blog/neurips_dataset>；
- 官方公开代码：<https://github.com/cam-mobsys/covid19-sounds-neurips>。

## 为什么不是直接跑一个模型

流程分成两道一次性命令：

1. **数据门**：只读取患者ID、标签、metadata、官方split和波形QC；不加载模型；
2. **正式模型**：只有数据门 `GO` 且研究者人工复核后才解锁。

这样可以保证 Cambridge 不是为了得到一个好结果才事后改 subset、matching 或字段。

## 合作者最快使用方式

```bash
cd external_validation/cambridge_covid_sounds
cp config.example.json config.local.json
# 只修改本地路径和原始CSV→标准字段的映射，不改阈值、arms、seeds或epoch
```

先运行数据门：

```bash
bash run_once.sh config.local.json gate
```

查看：

```text
output/.../public/data_gate.json
output/.../public/DATA_GATE_REPORT.md
```

如果 `NO_GO`，流程自动停止，不生成 AST/OPERA embedding 或模型分数。把
`public/PUBLIC_RESULTS.zip` 返回即可。

如果 `GO`，由项目负责人核对 aggregate gate 后，把配置中的：

```json
"formal_unlock": "REVIEW_GATE_BEFORE_SETTING_THIS"
```

改成：

```json
"formal_unlock": "FROZEN_PROTOCOL_AND_DATA_GATE_GO"
```

然后一键跑完正式阶段：

```bash
bash run_once.sh config.local.json formal
```

它依次执行：

```text
固定Phi-2文本缓存
        ↓
AST-6L / OPERA-CT 患者级表示
        ↓
Correct / Within-label / Global
        × 5 seeds × 500 epochs
        ↓
profile retrieval + probes + Standard/matched COVID AUROC/NLL
        + metadata-only / raw-audio+metadata direct-fusion baselines
        ↓
仅聚合的 PUBLIC_RESULTS.zip
```

## 输入方式

推荐把 Cambridge 官方 task-2 患者表与完整 DTA metadata 合并成一个 CSV／XLSX；也可以在
配置中分别提供participant表和metadata表，由程序按患者ID合并。文件必须至少
包含配置中映射的：

```text
participant ID, COVID label, official fold,
age, sex, smoking, cough, fever, sore throat,
shortness of breath, asthma, other respiratory disease, platform/cohort
```

音频有两种接入方式：

1. `audio_manifest_csv`：每行 `participant ID / relative path / modality`；推荐；
2. `cambridge_task2` 扫描：兼容官方 `0426_EN_used_task2` 目录结构，只选择文件名含
   `cough` 的录音。

多个 cough 文件会先各自编码，再在 participant 内等权平均。一个 participant 永远只贡献
一个 contrastive loss、一个 prediction 和一个 bootstrap unit。

## 隐私边界

```text
output/private/
```

包含患者清单、音频路径、匹配对和逐患者预测，受 DTA 约束，不能提交 GitHub，也不能默认
发回项目负责人。

```text
output/public/
```

只包含：aggregate QC、样本量、平衡诊断、AUROC/NLL差值、置信区间和文件哈希。合作者只需
返回 `PUBLIC_RESULTS.zip`。若 DTA 连聚合结果也有限制，应先由本地数据管理员核准。

## 重要边界

- 100 pairs 是“允许运行”的最低数据门，不自动构成外部 null confirmation；
- `Correct−Within` 不显著不能解释为相等；只有整个CI进入预注册的效应区间，才能称为
  equivalence-compatible；
- Cambridge hidden scorer 若只返回总体分数，不能承担 profile retrieval、matched subset、
  probes 或患者级 bootstrap；本包只使用 DTA 释放的官方患者级 split；
- 不允许为了 GO 修改 `0.12 SMD`、`0.08 fine balance`、标签、主音频或 matching 字段。

详细协议见 [PREREGISTRATION_ZH.md](PREREGISTRATION_ZH.md)，英文执行说明见
[COLLABORATOR_RUNBOOK.md](COLLABORATOR_RUNBOOK.md)。
