# Transfer Repair v2 外部 pair-support 数据门结果

日期：2026-09-04  
预注册：`docs/TRANSFER_REPAIR_V2_EXTERNAL_GATE_PREREG_ZH.md`  
结果：**CODA TB NO_GO；Cambridge NO_GO；未生成 Repair v2 embedding 或 prediction。**

## 一句话结论

> CODA TB 和 Cambridge 的 source train 都有充足的“同病、跨环境”positive，但没有足够的
> “同环境、异病、协变量可比”negative。问题不在找不到跨域同病样本，而在观察数据缺少同域
> 内可比较的异病对照；因此不能按冻结协议训练 Repair v2。

## 1. 三个数据集的统一视图

| 数据集 | source train N | cross-domain positive OK | strict negative OK | joint eligible | 各环境均有双标签 |
|---|---:|---:|---:|---:|---|
| UKCOVID | 20,714 | 36.06% | 27.81% | **0.246%** | 否 |
| CODA TB | 603 | 100.00% | 67.50% | **67.50%** | 是 |
| Cambridge | 551 | 100.00% | 45.37% | **45.37%** | 否 |

三者均低于事前冻结的总体与逐标签 80% joint-coverage 门槛。UKCOVID 是 source--label 几乎完全
绑定；CODA 和 Cambridge 则改善了跨域 positive 支持，但严格 matched negative 仍不足。

## 2. CODA TB：NO_GO

CODA source train 有 603 人（TB− 476、TB+ 127），七个 country 都同时包含 TB−／TB+：

| country | TB− | TB+ |
|---|---:|---:|
| IN | 70 | 6 |
| MG | 40 | 35 |
| PH | 114 | 7 |
| SA | 66 | 11 |
| TZ | 45 | 4 |
| UG | 95 | 40 |
| VN | 46 | 24 |

所有人都有至少两个同 TB、跨 country positive。失败发生在同 country、异 TB、同性别、年龄差
不超过10年且七项临床 Hamming distance 不超过2的 negative：

| population | N | positive OK | strict negative OK | joint eligible |
|---|---:|---:|---:|---:|
| 全部 train | 603 | 100.00% | 67.50% | **67.50%** |
| TB− | 476 | 100.00% | 60.50% | **60.50%** |
| TB+ | 127 | 100.00% | 93.70% | **93.70%** |

578/603 人至少存在一个同性别、同 country 的异标签候选；最近候选的年龄差中位数为5.5年、
clinical Hamming 中位数为1，但严格 caliper 下仍有196人没有任何合法 negative。因此这不是
“完全没有 overlap”，而是覆盖不足以让同一个 objective 代表完整训练人群。

## 3. Cambridge：NO_GO

Cambridge source train 有 551 人（COVID− 280、COVID+ 271）：

| platform | COVID− | COVID+ |
|---|---:|---:|
| ANDROID | 56 | 95 |
| IOS | 224 | 166 |
| WEB | 0 | 10 |

所有人都有至少两个同 COVID、跨 platform positive。但在同 platform、异 COVID，并对年龄、
性别和六项症状精确匹配后：

| population | N | positive OK | strict negative OK | joint eligible |
|---|---:|---:|---:|---:|
| 全部 train | 551 | 100.00% | 45.37% | **45.37%** |
| COVID− | 280 | 100.00% | 55.36% | **55.36%** |
| COVID+ | 271 | 100.00% | 35.06% | **35.06%** |

541/551 人在同 platform 至少有异标签参与者，但最近者在八个 exact fields 上的差异中位数是1；
只有250人达到差异为0。WEB 还存在 0 个阴性／10个阳性的结构性空单元，所以“每个环境双标签”
也失败。

## 4. 这说明什么、不说明什么

说明：

- 三个现有 source train 都不支持预注册的严格 Repair v2 监督；
- CODA／Cambridge 的主要瓶颈是 matched negative coverage，不是 cross-domain positive；
- source／disease／patient covariates 的 joint support 必须在方法训练前测量，不能把不可构造的
  objective 当成模型失败。

不说明：

- Repair v2 方法已被证明无效；它根本没有训练；
- 80% 是自然常数；它只是本轮事前冻结的代表性门槛，不能在看到结果后修改；
- CODA 或 Cambridge 的既有 Correct／Within／Global audit 无效；那些实验回答 correspondence
  与 transfer，和本轮“能否监督 repair”是不同问题。

## 5. 决策

1. UKCOVID、CODA TB、Cambridge 的单数据集 Repair v2 均停止；
2. 不运行 S0／S1、AST 五 seed、OPERA／HeAR 或任何 matched-target Repair v2 评分；
3. 不放宽 matching、删除环境条件或只在 eligible minority 上训练后宣称全人群 repair；
4. 若未来继续，应另立方法项目，以多个 cohort 联合训练来补足 `environment × disease` 单元，
   并保留一个新的、未用于设计的外部测试集。

## 6. 可审计文件

- CODA aggregate result：`results/transfer_repair_v2_coda_pair_gate.json`  
  SHA-256：`b21cf3b6c7581194aeced1f12fb466f7f717d5ec90e8d643bd5611adb46e002a`
- Cambridge aggregate result：`results/transfer_repair_v2_cambridge_pair_gate.json`  
  SHA-256：`a87392b88c6cbb1252f9ff3fc92d4daad89dfa67c389f290595695edbf37bf2c`
- gate implementation：`src/repair_v2_external_pair_gate.py`
- UKCOVID outcome：`docs/TRANSFER_REPAIR_V2_GATE_OUTCOME_ZH.md`

两份公开 JSON 均不含 participant ID；运行过程中未读取任何音频、表示、预测或 target model
score。
