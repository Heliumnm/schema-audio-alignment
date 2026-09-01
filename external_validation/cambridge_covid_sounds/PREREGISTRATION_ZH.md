# Cambridge COVID-19 Sounds 外部 Pairing-Controlled Transfer Audit 预注册

状态：**代码冻结候选；模型执行必须等待 DTA 环境中的数据门 GO。**

## 1. 科学问题

UKCOVID discovery audit 发现：正确音频--metadata 配对能学习逐患者 correspondence，但没有
稳定产生跨 backbone／readout、超过 raw audio 的 matched disease-transfer 收益。

Cambridge 分支只问：

> 在另一个患者级 COVID 音频数据集上，correct pairing 是否先产生可测 correspondence；
> 这种 correspondence 是否转化成协变量平衡后的疾病收益？

它不以复现某个绝对 AUROC 为目标，不与 UKCOVID 合并估计。

## 2. 数据与分析单位

- 数据：Cambridge COVID-19 Sounds NeurIPS 2021 DTA release；
- primary modality：`cough`；breathing/voice 不替代缺失 cough；
- unit：participant；所有 split、loss、prediction、matching 和 bootstrap 都是患者级；
- 多段 cough：每段先编码，participant 内等权平均；每人一个 loss；
- label：官方 task-2 COVID positive/negative；其他／未知状态排除；
- split：官方患者级 train/validation/test，不重新随机划分。
- 原始输入：官方 `data_0426_en_task2.csv` 的 `uid/label/fold`、完整 `covid19` 音频母目录
  （结构为 `ID/采集时间/audio_file_cough.wav`，亦兼容 `0426_EN_used_task2`）和
  `covid19/metadata` 下的 `android.csv / ios.csv / web.csv`；扫描仅进入 Task-2 ID，不使用
  Task 1。三份 metadata 为分号分隔并带空导出索引列，解析器按表头自动识别，不改写原始数据。

Cambridge hidden scoring queue 不承担本 audit，除非数据方明确提供逐患者、可匹配、可做配对CI
的输出。总体 leaderboard 分数只能是独立的 secondary benchmark。

## 3. 模型盲数据门

数据门允许读取：患者ID、COVID标签、官方split、metadata、音频路径和客观波形QC。禁止读取或
生成 embedding、projector、retrieval、AUROC、NLL 或模型预测。

### 3.1 Audio QC

- 可完整解码；
- finite；
- 非全零；
- 时长至少0.5秒；
- 保存 raw-file 和 canonical decoded-PCM SHA-256；
- 跨患者相同PCM为泄漏组，只按患者ID固定哈希保留一人，规则不读取标签。

### 3.2 Metadata schema

进入alignment文本：官方年龄段、sex、smoking、cough、fever、sore throat、shortness of breath、
asthma、other respiratory disease。缺失为 `[MISSING]`；未知枚举必须报错，不能默认为NO。

原始 metadata 适配规则在模型执行前冻结：年龄历史 typo `30-29` 规范为 `30-39`，德语
`Unter 20`规范为`00-19`；静态年龄段、
sex、smoking 在同一患者多行中不一致则排除；每日 `Symptoms` 和 `Medhistory` 采用
ever-positive 聚合（任一记录出现目标代码为YES；存在有效回答但无目标代码为NO；仅缺失／不愿
回答为`[MISSING]`）。platform优先由 Android/iOS/Web metadata 文件来源确定，文件名无法确定时
才使用官方Task-2 loader／数据字典的UID规则。以上处理不读取标签分布或模型结果。

不进入文本：COVID label、participant ID、audio ID、platform/site/location/device。这些只用作
matching、diagnostic或probe。

### 3.3 Matched endpoint

只在官方 test 中1:1、无放回匹配。exact字段：官方年龄段、sex、cough、fever、sore throat、
shortness of breath、asthma、other respiratory disease。原始数据只提供年龄段，因此不虚构
连续年龄；最小代价只辅助匹配smoking和platform，所有tie由固定SHA-256解决。

若最大匹配集不平衡，沿唯一的确定性greedy trimming路径逐对删除，第一次通过即停止；不得搜索
另一条更有利路径，也不得删除到100对以下。

### 3.4 GO／NO-GO

必须同时满足：

1. 至少100个matched positive/negative pairs；
2. 所有连续／二分类审计变量 `max |SMD| <= 0.12`；
3. platform和smoking的最大level比例差 `<=0.08`；
4. train至少500人且每类至少100；
5. validation每类至少30；test每类至少100；
6. participant和PCM group不跨split；
7. participant ID、metadata、label、split和cough linkage无歧义。

任一失败：`NO_GO`，正式模型阶段关闭。这是数据可行性结论，不是模型负结果。

官方Task 2约1,000名参与者，test约20%；100对等于200名参与者，因此该门几乎没有QC与exact
matching余量。这个紧约束在看数据门结果前保留；不得因为官方子集上限而事后降低门槛。

## 4. 冻结模型

- audio backbones：AST AudioSet前六层、OPERA-CT；均冻结；
- text encoder：Phi-2 revision `810d367871c1d460086d9f82db8696f2e0a0fcd0`，唯一schema只编码一次，右padding，mask-aware mean；
- projector：`768 -> 1024 -> 2560`，两层后LayerNorm+ReLU，第一层dropout 0.1；
- loss：单向 audio→text InfoNCE，temperature 0.07；
- optimiser：Adam lr 1e-3，默认参数，无scheduler/weight decay；
- batch 64，drop_last=False，每epoch重洗；
- five seeds `0..4`，500 epochs，只用最终checkpoint。

Arms：

- `correct`：同一participant；
- `within-label`：同COVID标签、不同participant、无self-pair；
- `global`：全局无self-pair；
- `raw_audio`：冻结音频表示参照。

另外固定两个不参与pairing机制判决的context baseline：`metadata_only`使用与alignment完全相同的
schema字段（training vocabulary one-hot，未见值单列），`raw_audio_plus_metadata`为直接拼接。
它们回答metadata本身的疾病预测力，以及不做alignment的late/direct fusion增量。

同seed的三个projector arm共享初始化、batch order和dropout stream，只有pairing不同。

## 5. 下游协议

- train拟合projector和下游head；
- validation用固定五折 one-SE 规则选logistic C，并拟合唯一source Platt calibrator；
- 官方test和matched subset不选择epoch、projector、C、readout或calibrator；
- test评价source-distribution，matched评价covariate-balanced endpoint；
- matched CI以matching pair为cluster，并在seed维保持配对。

## 6. 指标

### Correspondence

validation unique-profile macro MRR。主差值：`C-W`，profile×seed paired bootstrap。

### Information channels

至少报告sex、age 70+（与已发布年龄段边界一致）、cough、asthma和web-vs-app platform；若DTA字段支持，再加site/device。主差值为
matched `C-W`，`C-R`用于避免把“相对within保留”误写成“比raw编码更多”。

### Disease transfer

- calibrated NLL、AUROC、Brier、calibration slope；
- 主机制差：`C-W`；
- 方法收益边界：`C-R`；
- source test与matched分别报告，不合并。
- 同表报告`metadata_only`及`raw_audio_plus_metadata`；`RM-M`只解释为直接融合增量，不是
  alignment收益。

预定义最小有意义区间：`|Delta AUROC|<0.02`、`|Delta NLL|<0.01`。100 pairs只允许运行；
只有整个CI进入区间才称为equivalence-compatible。CI跨越有意义正负效应时必须写inconclusive。

## 7. 结果分支

1. retrieval `C-W>0`且CI排除0，disease `C-W`和`C-R`均稳定为正：TB/UKCOVID之外的
   metadata alignment可能有疾病收益；
2. retrieval通过，disease CI完全进入ROPE：支持correspondence--transfer gap外部复现；
3. retrieval通过但disease CI宽：外部结果不确定；
4. retrieval不通过：alignment manipulation失败，disease结果不可作机制解释；
5. `C-W>0`但`C-R<=0`：不能称为新方法收益；
6. backbone方向不同：不能声称backbone-robust external replication。

## 8. 受限数据输出

`private/`保存行级manifest、路径、pair和prediction，不离开DTA环境。`public/`只保存聚合计数、
平衡诊断、指标、CI和哈希。合作方默认只返回`PUBLIC_RESULTS.zip`；是否允许返回任何派生表征或
逐患者预测必须由DTA另行确认。

## 9. 禁止的修改

模型执行后不得：换标签、换主模态、修改SMD/fine-balance门槛、改matching字段、选择更有利的
platform/subset、增加seed/epoch/模型、用test挑calibrator、把不显著写成相等。
