# Cambridge COVID-19 Sounds：一键受限数据外部验证包

> **当前状态（2026-09-03）：旧match-first v2数据门已superseded；正式模型暂停。** 身份审计
> 发现旧代码把18个Web submission的恒定`Uid=form-app-users`合并为一人。正确Task-2结构为
> 1,000个subject，严格队列应为989人（500阴性、489阳性），不是旧gate的975人。模型尚未
> 运行，因此没有模型结果被污染。先看
> [CAMBRIDGE_TASK2_IDENTITY_AUDIT_ZH.md](CAMBRIDGE_TASK2_IDENTITY_AUDIT_ZH.md)，再由合作者用
> 修正版本重跑真实WAV数据门。

当前 `run.sh` 已增加identity-version guard；旧975人config会在加载模型前被明确拒绝。只有
修正数据门重新运行并经人工复核为`GO`后，英国合作者才可在本目录运行 `sbatch run.sh`。脚本使用
`/home/yl809/rds/hpc-work/datasets/covid19`、新 `cambridge_audit_output_webfix_v2`、合作者已经
准备好的 `/home/yl809/rds/hpc-work/env_audit` 和 RDS 模型缓存，一次完成正式阶段。脚本不再
读取 shell 启动文件、不依赖隐式 home 路径，也不会在作业中安装或替换 package。OPERA 的
顶层包名恰好也叫 `src`；当前代码会清除同名缓存、把 OPERA 根目录置于首位并验证实际导入
位置，避免把本项目的 `src` 误当成 OPERA。内部 runner 同样要求显式的 `PYTHON_BIN` 绝对
路径，所有 Python 脚本都由该解释器通过绝对路径调用，不再依赖裸 `python` 或继承的
`PYTHONPATH`。`csd3:` 前缀只用于从其他机器执行 scp/rsync，
不能写进 Slurm 内部文件路径。

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

### 当前推荐：修正Web namespace后重跑Task-2 match-first（先匹配target，再划开发集）

`structure.json` 正确展开为300个Android UID、682个iOS UID和18个Web Folder Name，共
1,000个subject、1,486次采集；每次均含breath、cough和voice/read。严格英语/标签规则在
structure+metadata层得到989人（500阴性、489阳性）。旧975人数字来自Web identity bug，
旧target与split不得复用。

```bash
bash run_reconstructed_match_first_v2.sh \
  /DTA/covid19/metadata \
  /DTA/test2 \
  /DTA/cambridge_match_first_v2
```

如果Task-1子集已经存在，可把它作为可选第四个参数加入，只用于跨任务provenance；本项目没有
使用Task 1，因此其缺失记录为未测量且不阻塞。程序先冻结恰好100对
`matched_target`，再把剩余人按`label × platform`固定划为70% train、15% validation、
15% source-test；随后生成不含ID的`public/overlap_report.json`。全程不加载模型。

新运行必须保留相同标签、matching字段、100对门槛、SMD 0.12与fine-balance 0.08；唯一修正
是Web participant namespace。结果与边界见
[CAMBRIDGE_TASK2_IDENTITY_AUDIT_ZH.md](CAMBRIDGE_TASK2_IDENTITY_AUDIT_ZH.md) 与冻结的
[PREREGISTRATION_MATCH_FIRST_V2_1_IDENTITY_FIX_ZH.md](PREREGISTRATION_MATCH_FIRST_V2_1_IDENTITY_FIX_ZH.md)。旧
[PREREGISTRATION_MATCH_FIRST_V2_ZH.md](PREREGISTRATION_MATCH_FIRST_V2_ZH.md) 与
[CAMBRIDGE_MATCH_FIRST_V2_GATE_OUTCOME_ZH.md](CAMBRIDGE_MATCH_FIRST_V2_GATE_OUTCOME_ZH.md)
保留为superseded历史。

### 历史v1：先随机划分，再只在test匹配

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
标记 `official_split_reproduced: false`，论文中必须称为 *reconstructed cohort*。这个v1路径
已经因test内不足100对而`NO_GO`；保留它是为了让版本变化可见，不应用它替代上面的v2命令。

#### 只有 Task-2 UID 列表时先做什么

先把 UID 列表与三平台 metadata 私下连接：

```bash
python src/match_task2_uid_metadata.py \
  --uid-list /private/UID.txt \
  --metadata-root /private/all_metadata \
  --output-root output/task2_uid_metadata_audit
```

逐行结果只写入 git-ignored 的 `output/.../private/`，可分享的聚合计数写入
`output/.../public/`。UID 列表只能确认 Task-2 成员；对多次填写问卷的参与者，它不能确定
Task-2 音频属于哪个采集时间。因此在冻结标签和 split 前，仍需 `UID + Folder Name`，或者让
重建脚本直接对 Task-2 音频树执行精确 session linkage。UID 和私有连接表不得提交 GitHub。

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

Slurm 提交入口为 `submit_formal.slurm`；模型、环境和私有配置的完整中文准备步骤见
`CAMBRIDGE_MODEL_ENV_SETUP_ZH.txt`。

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

## 为什么需要显式区分 v1 和 v2

官方论文报告的 Task 2 约为 1,000 名参与者，官方 test 约占 20%。当前预注册要求至少
`100 positive/negative matched pairs`，也就是 test 中至少 200 名可用且能通过 exact matching
的参与者。因此v1几乎没有QC或common-support余量，实际只得到26对。代码没有降低100-pair、
0.12 SMD或0.08 fine-balance阈值。v2也没有降低门槛，而是明确改变estimand：先从全部可用
Task-2人群冻结100对平衡target，再只用剩余人开发模型。它能回答secondary matched sensitivity，
不能恢复缺失的官方split，也不能成为原始confirmatory test。

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
[MISSING_TASK2_SPLIT_DECISION_ZH.md](MISSING_TASK2_SPLIT_DECISION_ZH.md)。集群提交说明见
[SLURM_GATE_GUIDE_ZH.txt](SLURM_GATE_GUIDE_ZH.txt)，可直接提交的脚本为
[submit_reconstructed_gate.slurm](submit_reconstructed_gate.slurm)（v1）或
[submit_reconstructed_match_first_v2.slurm](submit_reconstructed_match_first_v2.slurm)（当前v2）。
