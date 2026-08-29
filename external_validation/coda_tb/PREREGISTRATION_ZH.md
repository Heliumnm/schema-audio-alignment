# CODA TB 外部验证：数据可行性预注册

冻结日期：2026-08-29  
数据入口：Synapse `syn39711065`，solicited cough `syn40358494`

## 1. 问题与边界

本阶段只回答：CODA TB 能否公平执行 UKCOVID 已定义的
Pairing-Controlled Transfer Audit？它不是模型实验，也不允许产生 AST/OPERA
embedding、alignment checkpoint、TB 模型分数、AUROC、NLL 或 retrieval 指标。

受控数据、participant ID、音频路径、匹配对和 embedding 不进入 GitHub。公开仓库只保留
代码、预注册和经过 disclosure check 的聚合结果。

## 2. 数据单位和标签

- 单位是 participant，不是 cough file；
- primary audio 只用 clinic-supervised solicited cough；
- participant 至少有一条通过客观 QC 的 solicited WAV 才可进入；
- primary label 是 `tb_status`，即官方 microbiologic reference standard；
- longitudinal cough 不替代 solicited cough，也不用于挽救样本量；
- `sound_prediction_score` 完全忽略，不参与 QC、matching 或 GO/NO-GO。

## 3. 客观音频 QC

每个映射文件都必须存在并可完整 PCM 解码。文件级纳入要求：普通非空文件、WAV header
与 decoded frame 一致、有限且非全零、时长至少 0.20 秒。采样率、声道数、位深和时长
只按统一规则审计，不按 TB label 设置不同阈值。

同一 participant 内的相同 decoded PCM 只保留一份。跨 participant 的相同 PCM 由
`sha256("coda-dedup-v1|participant")` 最小者拥有；其他 participant 只移除该重复文件，
仍有独立有效 cough 时可以保留。跨 participant 的 raw/PCM duplicate 数必须报告，且任何
重复 PCM 不得跨 split。

## 4. Metadata schema

允许进入未来 alignment text 的字段族预先固定为：age、sex、height、weight、
reported cough duration、prior TB/type、hemoptysis、heart rate、temperature、weight loss、
smoking in the last week、fever、night sweats。缺失值显式写为 `[MISSING]`。

禁止进入 alignment text：TB label、country、site/device、participant ID、
microbiologic/Xpert 结果、HIV status、采集文件数和 `sound_prediction_score`。
Country 和 HIV 只作为 domain/clinical audit covariates。

逐字段报告 overall、TB+/TB−、country 条件下的 missingness。缺失本身可能是 country
proxy；本阶段不得因为看到匹配结果而临时增删 schema 字段。

## 5. 冻结 split

在音频 QC 和跨 ID duplicate 处理后，以 `label × country × sex` 分层。每层 participant
按 `sha256("coda-outer-v1|participant")` 排序，前 60% 进入 development candidate，后
40% 进入 untouched target candidate。Target candidate 无论最后是否被匹配，都不得回流
development。

Development 再按同一分层和 `sha256("coda-dev-v1|participant")` 固定为 70% train、
15% validation、15% source-test。Matched target 从来不选择 epoch、projector、classifier、
regularisation 或 calibration。

## 6. 冻结 matching

Target candidate 中进行 1:1、无放回、country × sex 精确匹配。每个 exact stratum 内用
确定性最小成本二分图匹配；成本只使用预先固定的 patient covariates：

- age difference / 10 years；
- height difference / 10 cm；
- weight difference / 10 kg；
- `|log1p(cough_days_a)-log1p(cough_days_b)|`；
- heart-rate difference / 10 bpm；
- temperature difference / 1 °C；
- prior TB、prior TB type、hemoptysis、weight loss、recent smoking、fever、night sweats、
  HIV 的每项 mismatch 各 1；
- tie 由 `sha256("coda-match-v1|negative|positive")` 唯一打破。

先取 maximum-cardinality assignment。若 balance 未通过且仍多于 100 对，则沿唯一的
greedy trimming path 每次删除一对：选择使下列最大标准化违规量最小的删除；平局由 pair
hash 打破。第一次通过即停止；不得低于 100 对，也不得另搜一条更有利的 trimming path。

## 7. GO/NO-GO

全部满足才 GO：

1. clinical 与 additional 表均为 1,105 个唯一 participant，集合一致；
2. solicited 映射为 9,772 个唯一 filename，所有映射 participant 均存在；
3. 下载文件集合与映射逐项一致，无缺失、无额外 WAV；
4. QC 后至少 1,000 个 participant，至少 250 TB+ 和 700 TB−；
5. matched target 至少 100 对；
6. country 与 sex exact balance；所有 numeric/binary dummy 的最大绝对 SMD ≤0.12；
7. country、sex、prior TB、HIV 和各症状 level 的最大比例差 ≤0.08；
8. development train ≥400 且两类各 ≥100；validation/source-test 各 ≥75 且两类各 ≥15；
9. primary schema 至少 100 个 unique profile，最大 profile 不超过 eligible participants
   的 10%；
10. split participant overlap 为 0，decoded PCM duplicate 不跨 split；
11. 本阶段没有加载任何 audio encoder、representation、prediction 或 disease score。

任一 primary gate 失败即：`NO-GO for confirmatory disease-transfer analysis`。不得改变
modality、split、matching covariates、100-pair floor、0.12 SMD 或 0.08 fine-balance 门槛。

## 8. GO 后才允许的实验

GO 后另行冻结并执行 frozen AST/OPERA → Correct / Within-label / Global × 5 seeds，依次
报告 profile retrieval、patient/clinical/cohort/recording probes、source 与 matched TB
AUROC、NLL、Brier 和 calibration。核心比较仍是：

- Correspondence Gain：Correct − Within-label retrieval；
- Transfer Gain：Correct − Within-label matched calibrated disease performance。

CODA 是独立 external stress test；不与 UKCOVID 数值池化。任何正结果仍按受控训练集上的
内部 target 解释，不冒充官方 CODA challenge validation。

