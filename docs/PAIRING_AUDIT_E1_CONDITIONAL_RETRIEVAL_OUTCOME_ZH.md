# E1 条件化 profile retrieval：正式结果

日期：2026-09-11

性质：**看到主结果后预注册的定向诊断；不是独立确认实验。**

冻结协议：`docs/PAIRING_AUDIT_SUPPLEMENTARY_EXPERIMENTS_PREREG_ZH.md`

## 1. 这一步回答什么

UKCOVID 主分析发现，Correct pairing 相对 Within-label pairing 有稳定但很小的 profile retrieval
优势。本诊断不重新训练模型，只改变检索时允许进入候选库的 profile，检查该优势是否主要依赖
性别或疾病标签提供的粗粒度候选差异。

本分析只读取 Standard validation。matched、matched-long 和 test 标签均未读取。

## 2. 候选库审计

所有条件均保留 5,179 个 query 和 1,871 个真实 profile，真实 profile 覆盖率为 100%。候选资格在
Correct、Within-label、Global 以及五个 seed 之间逐位相同；相同 schema 只算一个语义候选。

| 候选条件 | 每个 query 的候选数范围 | 中位候选数 | analytic random macro-MRR |
|---|---:|---:|---:|
| unrestricted | 1,871 | 1,871 | 0.00434 |
| same sex | 725--1,146 | 1,146 | 0.00790 |
| same label | 476--1,507 | 476 | 0.00729 |
| same label + sex | 206--943 | 270 | 0.01317 |

候选库大小不同，因此不同条件的原始 MRR 不能直接相减，也不能把 MRR 变化比例解释成某个字段的
“贡献百分比”。正式比较始终是在同一候选条件内、逐 profile × seed 配对的 Correct−Within。

## 3. AST-6L

| 候选条件 | Correct MRR | Within MRR | Correct−Within，95% CI | 五 seed 方向 |
|---|---:|---:|---:|---|
| unrestricted | 0.00891 | 0.00574 | **+0.00318 [0.00161, 0.00492]** | 5/5 正 |
| same sex | 0.01159 | 0.01019 | +0.00140 [−0.00065, 0.00350] | 5/5 正 |
| same label | 0.01310 | 0.00840 | **+0.00470 [0.00283, 0.00666]** | 5/5 正 |
| same label + sex | 0.01751 | 0.01506 | +0.00245 [−0.00001, 0.00499] | 5/5 正 |

## 4. OPERA-CT

| 候选条件 | Correct MRR | Within MRR | Correct−Within，95% CI | 五 seed 方向 |
|---|---:|---:|---:|---|
| unrestricted | 0.00877 | 0.00554 | **+0.00323 [0.00155, 0.00500]** | 5/5 正 |
| same sex | 0.01058 | 0.00998 | +0.00060 [−0.00143, 0.00256] | 2 正／1 近零／2 负 |
| same label | 0.01265 | 0.00807 | **+0.00459 [0.00256, 0.00670]** | 5/5 正 |
| same label + sex | 0.01565 | 0.01471 | +0.00094 [−0.00149, 0.00329] | 3 正／2 负 |

置信区间为冻结的 10,000 次 profile × seed 配对 bootstrap，RNG `20260820`。

## 5. 结论

疾病标签的粗粒度候选结构不能解释原 retrieval 结果：限制为同标签候选后，两个 backbone 的
Correct−Within 仍为稳定正值。

性别条件则明显改变两个 primary backbone 的结论：限制为同性别以后，AST／OPERA 的效应都
缩小且置信区间跨零；同时限制标签与性别时也没有得到二者稳定区间。因此，本诊断支持以下有限表述：

> UKCOVID 中原有的 exact-pair profile-retrieval 优势对候选库的性别结构高度敏感；粗粒度疾病
> 标签不足以解释该优势，但控制记录性别后，剩余优势不再稳定。

这与主分析中 Correct−Within 的 sex decodability 明显为正相互吻合，但不能证明疾病分类器在因果
意义上使用了性别，也不能把缩小幅度解释为“性别贡献比例”。

### 5.1 事后 HeAR 扩展

按独立冻结的事后协议，HeAR 后续也完成 E1：

| 候选条件 | Correct−Within macro-MRR，95% CI |
|---|---:|
| unrestricted | +0.00711 [0.00476, 0.00971] |
| same sex | +0.00356 [0.00096, 0.00638] |
| same label | +0.00923 [0.00623, 0.01240] |
| same label + sex | +0.00451 [0.00115, 0.00813] |

HeAR 在性别限制后同样缩小，但仍排除 0；因此 E1 最终说明结果对性别候选结构敏感，而不是说明
它可以解释所有 backbone 的全部 retrieval 差异。HeAR 是事后稳健性诊断，不改变 AST／OPERA
作为 primary backbones 的地位。

## 6. 对下一步的影响（现已执行）

E1 没有推翻“pairing intervention 产生可测表示差异”，但缩小了它的解释范围。按照事前顺序，下一步
执行 (W_{y,s})：在训练时同时保留疾病标签与记录性别，再打乱 participant pairing。该实验现已在
AST、OPERA 和事后 HeAR 上完成；三个 backbone 的 `C-W_{y,s}` sex probe 均跨零。完整 E2/E3
结果见 `docs/PAIRING_AUDIT_E1_E2_E3_OUTCOME_ZH.md`。

## 7. 可复现文件

- 实现：`src/eval_conditional_profile_retrieval.py`
- AST 聚合结果：`results/pairing_followup/e1_ast/metrics.json`
- OPERA 聚合结果：`results/pairing_followup/e1_opera_ct/metrics.json`
- HeAR 聚合结果：`results/pairing_followup/e1_hear/metrics.json`
- 运行日志：`results/pairing_followup/logs/e1_ast.log`、`e1_opera_ct.log`
- 含 participant ID 的逐 query rank 文件只保存在本地和服务器，不发布到公开仓库；其 SHA-256
  前 16 位为 AST `f5f3b24820ecc258`、OPERA `b99b0cd88f01433d`
- 两条运行使用同一脚本，脚本 SHA-256 前 16 位：`b33e7487d9c71d5c`
