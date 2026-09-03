# Cambridge “Task 2” 身份核对：它到底是什么？

日期：2026-09-03  
性质：只读取官方说明、目录结构、metadata 与 participant identifier；没有读取任何模型输出。

## 最终结论

> 数据来源是 **Cambridge COVID-19 Sounds NeurIPS 2021 的官方 Task 2 音频子集**；官方
> Task 2 的任务是 COVID 阳性与检测阴性分类。我们当前执行的不是缺失的官方 split 复现，
> 而是在这个 Task-2 子集上自行冻结的、更严格的 COVID endpoint：英语、近14天阳性
> (`last14`／`positiveLast14`) 对从未阳性的检测阴性 (`negativeNever`)，并且只用 cough。

因此三个候选答案应这样判：

| 候选解释 | 判定 | 原因 |
|---|---|---|
| UKCOVID Task 2 | **不是** | 两套数据来源、字段和 participant namespace 不同；ID 集合零重叠 |
| COVID-19 Sounds custom COVID task | **是，准确描述当前 endpoint** | 来源是官方 Task-2 子集，但标签筛选、split、matching 是本项目重建并冻结的 |
| COVID-19 Sounds symptom task | **不是** | 症状预测是官方 Task 1；Task 2 明确是 COVID prediction |

一句最不容易被审稿人误解的名称是：

> **Cambridge COVID-19 Sounds Task-2-subset, reconstructed strict-COVID cough audit**

不能写成 `UKCOVID Task 2`，也不能写成 `official Cambridge Task-2 replication`。

## 1. 原始 dataset root

受限发布的母目录包含：

- `all_metadata/`：三平台完整 metadata；
- `covid-19data/`：完整 200GB+ 音频母库；
- `task1/`：论文 Task 1 子集；
- `task2/`：论文 Task 2 子集。

本次 `structure.json` 的根节点名称是 `test2`，内容与 Task-2 子集的官方规模完全吻合。
合作者环境中当前挂载／扫描路径是：

```text
/home/yl809/rds/hpc-work/datasets/covid19
```

这只是合作者给 Task-2 子集选择的本地目录名，不是 UKCOVID 的 dataset root，也不是完整
`covid-19data/` 母库。

## 2. participant ID namespace

目录结构应按官方 loader 的三平台规则解释：

| 平台 | Task-2 subject key | 数量 |
|---|---|---:|
| Android | 10字符 `Uid` | 300 |
| iOS | 12字符 `Uid` | 682 |
| Web | `form-app-users/<Folder Name>` 中的 `Folder Name` | 18 |
| **合计** |  | **1,000** |

根目录还含一个名为 `metadata` 的非参与者目录，不能计入 participant。`UID.txt` 只是根目录
条目列表：它把 `metadata` 和 `form-app-users` 各算成一个值，却没有展开18个 Web folder，
所以其984行不能当作正式 subject list。

这一点暴露了旧重建代码的错误：三份 all-metadata CSV 中 Web 的 `Uid` 恒为
`form-app-users`；旧代码据此把18个 Web 提交合并成一人。官方 Task-2 loader 实际使用
`Folder Name` 读取 `form-app-users/<folder>`，所以 Web 必须按 folder 分开。

## 3. task label definition

Cambridge 官方仓库定义：

- Task 1：是否报告呼吸道症状；
- Task 2：COVID 阳性与检测阴性分类；阳性可以无症状，阴性也可以有非 COVID 呼吸道症状。

官方入口是 `data_0426_en_task2.csv` 与 `0426_EN_used_task2/`。当前受限副本缺少前者，因而
无法恢复官方成员、标签与 fold。本项目才另行冻结了严格重建标签：

- positive：`last14`、`positiveLast14`；
- negative：`negativeNever`；
- 其余状态排除；
- 同一 subject 若出现正负冲突则整人排除；
- 多次合格采集只以标签盲 hash 选择一次。

所以它仍是 COVID 任务，但不是原论文 Task-2 split 的复现。

官方来源：

- <https://github.com/cam-mobsys/covid19-sounds-neurips#task-2-covid-19-prediction>
- <https://github.com/cam-mobsys/covid19-sounds-neurips/blob/master/COVID19_prediction/data/pickle_data.py>
- <https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/e2c0be24560d78c5e599c2a9c9d0bbd2-Abstract-round2.html>

## 4. audio modality

目录中1,486次采集全部同时具有：

- breath；
- cough；
- voice/read。

官方 Task-2 `pickle_data.py` 也同时载入三种模态。当前 external audit 为了与 UKCOVID／CODA
的 cough 口径一致，预先固定为 **cough-only**，并且每位进入分析的 subject 只保留一个
标签盲选择的合格 cough。不能把“Task 2 有三模态”和“本 audit 用 cough-only”混成一句话。

## 5. 与 UKCOVID participant ID 的 overlap

使用本地 UKCOVID `participant_metadata.csv` 的72,999个 participant identifier 与正确展开的
Cambridge 1,000个 Task-2 subject key 比较：

| 检查 | 重叠数 |
|---|---:|
| 原字符串精确相等 | **0** |
| 忽略大小写 | **0** |

UKCOVID ID 固定为 `P` 加7位数字；Cambridge 使用 Android 10字符 UID、iOS 12字符 UID，
以及 Web `Folder Name`。这证明它们是独立 participant namespace，不是同一数据集的两个任务。

## 6. 旧 Cambridge gate 为什么必须作废？

正确展开 Web namespace 后：

- Task-2：1,000 subjects、1,486 sessions；
- 1,482 / 1,486 session 可与 metadata 精确连接；
- 严格英语／COVID 标签后：**989 subjects = 500 negative + 489 positive**；
- label conflict：0。

旧代码把15个符合严格规则的 Web positive 合并成1人，因此报告为
`975 = 500 negative + 475 positive`。这不是统计波动，而是 participant identity bug。旧的
975人数据门与其100对 target 必须标记为 **superseded**；Cambridge 模型不得从旧 config 启动。

修复只改变 Web participant namespace，不修改标签、matching 字段、100对门槛、SMD 0.12、
fine-balance 0.08、AST／OPERA、seed 或 epoch。合作者必须用修复版本重新执行真实 WAV 数据门；
通过后再审阅新的 aggregate 输出并解锁模型。

机器可读汇总：`../../results/cambridge_task2_identity_audit.json`。
