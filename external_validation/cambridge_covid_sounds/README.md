# Cambridge COVID-19 Sounds：一键受限数据外部验证包

> **当前状态（2026-09-01）：等待有 DTA 授权的英国合作者执行。** Coswara 与 CODA TB
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

1. **数据门**：只读取患者ID、标签、metadata、冻结split和波形QC；不加载模型；
2. **正式模型**：只有数据门 `GO` 且研究者人工复核后才解锁。

这样可以保证 Cambridge 不是为了得到一个好结果才事后改 subset、matching 或字段。

## 合作者最快使用方式

### 当前实际发布包没有 Task-2 CSV：从原始 metadata 重建（当前推荐入口）

本次收到的受限数据中，`task1/` 和 `task2/` 只有音频，没有仓库说明里提到的
`data_0426_en_task2.csv`。因此不能声称复现官方 Task-2 成员和 split。代码提供一条独立、
模型盲态的重建路径：

```bash
bash run_reconstructed_gate.sh \
  /DTA/covid19/metadata \
  /DTA/covid19 \
  /DTA/cambridge_audit_output
```

其中第二个参数也可以指向仅含 Task-2 音频的目录。程序会：

- 通过 `Uid + Folder Name` 把一次采集的 metadata 与同次 cough 精确连接；
- 只保留英语、近14天阳性（`positiveLast14/last14`）和从未阳性的阴性
  （`negativeNever`）；
- 排除同一患者跨采集出现两种严格标签的情况；
- 每位患者用固定、与标签无关的哈希选择一个有 cough 的采集时点；
- 按患者、在 `label × platform` 内固定哈希切成 70%/10%/20%；
- 再执行同一个100对、0.12 SMD、0.08 fine-balance数据门。

选择 70/10/20 是因为官方论文使用这个比例；不是为了复原已经缺失的官方成员关系。输出会明确
标记 `official_split_reproduced: false`，论文中必须称为 *reconstructed cohort*。

### 如果以后找到官方 Task 2 CSV：官方成员与 split 路径

当前研究问题是 COVID-19 disease transfer，因此使用 **Task 2**，不是用于“是否有呼吸症状”的
Task 1。官方 Task 2 入口文件是：

```text
task2/data_0426_en_task2.csv       # 官方 uid / label / fold
covid19/                           # 完整音频母目录（也兼容 0426_EN_used_task2）
covid19/metadata/                  # android.csv / ios.csv / web.csv
```

只安装数据门依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gate.txt
```

然后用一条命令运行完全 model-blind 的检查：

```bash
bash run_task2_gate.sh \
  /DTA/task2/data_0426_en_task2.csv \
  /DTA/covid19/metadata \
  /DTA/covid19 \
  /DTA/cambridge_audit_output
```

脚本会自动：

- 按 `uid` 连接官方 Task-2 split 与三平台 metadata；
- 把官方年龄段（含历史 `30-29` typo）标准化；
- 把 `Symptoms` 和 `Medhistory` 多选字段拆成预注册字段；
- 从 metadata 文件来源／官方 UID 规则恢复 Android、iOS、Web；
- 只进入 Task-2 CSV 所列 ID 的目录，识别 `ID/采集时间/audio_file_cough.wav`，不会质检整库；
- 扫描 cough、执行客观波形 QC、做 test 内匹配和平衡检查；
- 若 `NO_GO`，打包聚合结果并在任何模型加载前停止。

本步骤的正式输出是：

```text
/DTA/cambridge_audit_output/public/DATA_GATE_REPORT.md
/DTA/cambridge_audit_output/public/data_gate.json
/DTA/cambridge_audit_output/public/PUBLIC_RESULTS.zip
```

### 已经手工整理成 canonical table（高级用法）

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

## 输入方式与冻结聚合规则

当前推荐用 `run_reconstructed_gate.sh` 从三平台metadata和按采集时间组织的音频树自动生成
冻结manifest。若以后找回官方CSV，再使用 `config.task2_raw.example.json`；程序会自动识别
Task-2表的逗号分隔，以及 `android.csv / ios.csv / web.csv` 的分号分隔和空导出索引列。
静态字段（年龄段、性别、吸烟）在同一患者的多行记录中必须一致，否则患者被排除；每日
`Symptoms` / `Medhistory` 使用事前固定的 **ever-positive** 聚合：任一记录出现目标代码为 YES，
有有效回答但没有目标代码为 NO，只有缺失／不愿回答为 `[MISSING]`。该规则不读取标签分布或
模型结果。

若使用高级 canonical-table 模式，文件必须至少包含配置中映射的：

```text
participant ID, COVID label, official fold,
age, sex, smoking, cough, fever, sore throat,
shortness of breath, asthma, other respiratory disease, platform/cohort
```

音频有两种接入方式：

1. `audio_manifest_csv`：每行 `participant ID / relative path / modality`；推荐；
2. `cambridge_task2` 扫描：兼容 `covid19/ID/采集时间/audio_file_cough.wav` 和官方
   `0426_EN_used_task2`；只进入 Task-2 CSV 中的 ID，并只选择文件名含 `cough` 的录音。

多个 cough 文件会先各自编码，再在 participant 内等权平均。一个 participant 永远只贡献
一个 contrastive loss、一个 prediction 和一个 bootstrap unit。

## 一个必须提前知道的数据门限制

官方论文报告的 Task 2 约为 1,000 名参与者，官方 test 约占 20%。当前预注册要求至少
`100 positive/negative matched pairs`，也就是 test 中至少 200 名可用且能通过 exact matching
的参与者。因此这个门几乎没有 QC 或 common-support 余量。代码不会为了让外部验证通过而降低
100-pair、0.12 SMD 或 0.08 fine-balance 阈值；若因此 `NO_GO`，正确结论是“该官方子集不足以
承担预注册的 confirmatory transfer endpoint”，不是模型失败。

对本机三份 metadata 的模型盲审计（音频尚未连接）得到：53,449行；英语且属于上述严格标签的
3,493位患者中，48位跨时间出现正负冲突；排除后为521位仅阳性、2,924位仅阴性。按20% test，
阳性理论上约104人，因此重建路径同样几乎没有QC和matching余量。这个数字解释门槛风险，不能
用来修改split或阈值。

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
[COLLABORATOR_RUNBOOK.md](COLLABORATOR_RUNBOOK.md)。可直接转发给合作者的中文纯文本快速指南见
[COLLABORATOR_QUICKSTART.txt](COLLABORATOR_QUICKSTART.txt)。官方split文件缺失后的证据和冻结决定见
[MISSING_TASK2_SPLIT_DECISION_ZH.md](MISSING_TASK2_SPLIT_DECISION_ZH.md)。
