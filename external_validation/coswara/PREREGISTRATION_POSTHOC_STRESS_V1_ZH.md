# Coswara 三骨干外部压力测试：冻结协议 v1

冻结日期：2026-09-03  
状态：**在读取任何 Coswara AST、OPERA、HeAR、projector、retrieval、probe 或 COVID 模型分数之前冻结**

## 1. 科学地位

Coswara 原预注册贪心数据门在 100 对下限时最大分类 SMD 为 0.1401，故正式外部确认仍是
`NO-GO`。之后进行的、完全不读取模型表示的全局约束敏感性分析找到了一个满足原阈值且更严格
细平衡约束的 100 对队列，判为 `SECONDARY_GO`。

因此本实验只能称为 **post-hoc external stress test（事后外部压力测试）**，不能称为 untouched
external confirmation，也不能覆盖原 `NO-GO`。它只问：在另一套 crowdsourced COVID 咳嗽数据
中，`Correct > Within-label` 的 participant correspondence 是否出现，以及该差异是否伴随平衡
人群中的 COVID transfer。

## 2. 冻结数据

- 仅用每位 participant 的一条 `cough-heavy.wav`；participant 是训练、评测和 bootstrap 单位；
- 客观 QC、PCM 去重、疾病标签定义、eligibility 和 matching 字段均沿用原 Coswara 数据门；
- 冻结 participant manifest：1,701 人；train 1,041、validation 217、source-test 243、
  matched-target 200（100 个正负 pair）；
- 三个受控输入 SHA-256：
  - audio QC：`e9d9c3578a8fdb0e14ffaaf73e2fe64cec1f01877a583b4ee4e9591f545bf4d7`；
  - participant manifest：`4f998a6bb7b1370b9b3f2905023ec170dd148a61a1767973d7d1c510fed92bea`；
  - matched pairs：`ffa995b61c34bd0f83e83b401686380d25b94a11301d129863fba9c7f279efd7`。

`matched-target` 从不选择 encoder、文本、projector、epoch、readout、正则、阈值或校准器。
source-test 和 matched-target 只在全部表示、训练和 validation 选择完成后统一读取。

## 3. Metadata 文本

字段顺序固定为：

`age, sex, cough, fever, fatigue, sore_throat, breathing_difficulty, asthma,
other_respiratory, smoker, diarrhoea, loss_of_smell, vaccination, mask_use`。

每项写成固定 typed schema token，例如 `[AGE=37] [SEX=male]`；missing 显式写 `[MISSING]`。
明确排除 COVID label/status、country/province、日期/halfyear、test status/type、participant ID、
录音质量与文件属性。不得生成 LLM reasoning caption。

冻结队列共有 1,156 个唯一 profile；最大 profile 21 人。validation 为 217 人、186 个唯一
profile，故 retrieval 必须 duplicate-aware，并对 profile 做 macro 平均，不能把同文本 participant
当成不同答案。

文本编码器沿用主实验的冻结 Phi-2 revision
`810d367871c1d460086d9f82db8696f2e0a0fcd0`，右 padding、mask-aware mean pooling，先实测
全量 token 长度再固定不截断的 max length；每条唯一文本只编码一次并按 text ID 复用。

## 4. 三个音频骨干

AST-6L、OPERA-CT 和 HeAR 各自独立执行同一队列、文本、pairing、readout 与评测。不得根据某个
backbone 的结果修改其他 backbone，也不得只报告最有利模型。

- AST-6L：16 kHz、10.24 s、只平均完全落在真实音频内的 patch，participant 输出 768 维；
- OPERA-CT：原生 16 kHz mel/静音处理与 8 s full-coverage 聚合，输出 768 维；
- HeAR：revision `9b2eb2853c426676255cc6ac5804b7f1fe8e563f`，16 kHz mono，2 s
  deterministic full coverage，短音频右补零，窗口等权平均，输出 512 维。

encoder 全冻结。每个 backbone 先做分层重复提取 preflight；正式缓存必须覆盖 1,701/1,701、
participant 顺序与 manifest 逐位一致、有限、非零。任何技术失败都停止，不能删除 participant。

## 5. Pairing、projector 与训练

仅在 train 内构造：

| arm | 音频的文本正样本 | 保留的信息 |
|---|---|---|
| `correct` | 自己的 metadata | label group + participant correspondence |
| `within_label` | 同 COVID label、不同 participant | label group，破坏 participant correspondence |
| `global` | 全 train 随机不同 participant | 破坏 label group 与 participant correspondence |

shuffle 是每 seed 固定一次的无固定点双射。seeds 固定为 0–4；同 backbone、同 seed 的三臂共享
初始化、batch 顺序和 dropout 随机流，唯一差异是 pairing。

projector 为 `input_dim -> 1024 -> 2560`；两层后均为 LayerNorm + ReLU，第一层后 dropout
0.1。单向 audio-to-text InfoNCE，temperature 0.07，batch 64，Adam lr 1e-3、默认参数、无
scheduler/weight decay，500 epochs，只用 epoch-500。音频与文本 encoder 均冻结。

## 6. 评测

### 6.1 Correspondence

在 validation 评估 duplicate-aware profile retrieval。candidate bank 是 186 个唯一 profile；同一
profile 的 participant 共用一个正确 text ID。主量为 macro-profile MRR 的 `correct-within`，
次要量为 micro MRR、R@1、R@10 和 `within-global`。profile × seed 配对 bootstrap 10,000 次；
只有主量 CI 下界大于 0 才认为 correspondence 建立。

### 6.2 COVID transfer

对 `raw/correct/within/global` 使用相同 logistic readout；另报 metadata-only、metadata+raw 和
metadata+三个 aligned representation。标准化器和 train-vocabulary metadata encoder 只在 train
拟合。C 网格固定为 `[.001,.003,.01,.03,.1,.3,1,3]`；在 validation 五折 one-SE 规则选最小 C；
Platt 只在 validation 拟合并原样应用到 source-test 与 matched-target，target 不重校准。

主要 transfer estimand 是 matched-target 的 `correct-within` 配对 `delta(-NLL)`；共同主判据还
要求 `delta AUROC` 为正且 CI 下界大于 0。主 CI 按 exact matching stratum × seed 分层配对
bootstrap 10,000 次；pair × seed bootstrap 为敏感性分析。另报 Brier、calibration slope/intercept、
source-test、`correct-raw`、`within-global` 与 direct fusion。

### 6.3 信息通道

同 source-only linear probe 比较 `raw/correct/within/global`：

- patient：sex、age >= 45；
- clinical context：cough、fever、fatigue、sore throat、breathing difficulty、asthma、other
  respiratory、smoking、diarrhoea、loss of smell；
- cohort（不进入对齐文本）：country group、halfyear；
- acquisition（不进入对齐文本）：duration、RMS、clipping、silence fraction、manual quality。

缺失 target 排除，不当作 negative；某项 source train/validation 缺少一类则报 `not_estimable`。

## 7. 解释分支

1. retrieval 不通过：alignment correspondence 未建立，不能解释 disease 差异；
2. retrieval 通过且 matched 两个 transfer 主量都通过：Coswara 压力测试中 correspondence 与
   transfer 同时提高，但仍不是 untouched confirmation；
3. retrieval 通过且两项 CI 完全落入预设等价区间（AUROC +/-0.02、delta(-NLL) +/-0.01）：支持
   correspondence/transfer 分离；
4. retrieval 通过但 CI 含 0 且不完全等价：`inconclusive`；
5. correct 显著更差：报告 negative transfer；
6. 三 backbone 不一致：报告 backbone-dependent，禁止挑最好结果。

## 8. 执行顺序与公开范围

先提交本协议，再实现和测试；随后 S0、三 backbone 提取、文本缓存、train-only S1、三臂五种子
正式训练、冻结 validation readout，最后一次性读取两个 test population。

原始音频、含 ID manifest、pairs、embedding、逐 participant prediction 和 checkpoint 均不进入
GitHub。仓库只提交代码、配置模板、输入/输出 hash、aggregate 指标、CI 与不含 ID 的日志。

