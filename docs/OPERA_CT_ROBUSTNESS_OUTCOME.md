# OPERA-CT 第二音频编码器稳健性结果

> **后续更新（2026-08-23）：** 本文记录 OPERA 的首轮线性评估；Route-A 已进一步补齐
> profile retrieval、information-channel probes、raw-preserving fusion、target calibration
> transport 和固定 MLP。最终综合结论见 `ICASSP_EXECUTION_STATUS_ZH.md`。

状态：**正式完成。** 这是同一 UKCOVID 发现性队列上的 backbone robustness，不是外部
确认。

## 为什么做这一步

AST-6L 的结果可能只属于一个通用音频编码器。这里把唯一变量换成面向呼吸音预训练的
OPERA-CT，其余内容全部保持不变：同一批参与者、同一 metadata 文本、同一 Phi-2 缓存、
同一 projector、correct／within-label／global 三种配对、五个 seed、500 epochs，以及同一
Standard-validation 选型和 source-domain 校准。

## 表征提取与训练审计

- `72,458 / 72,458` 名参与者全部成功，顺序与冻结队列逐位一致；
- OPERA-CT 输出为 `72,458 × 768`，全部有限、非零；
- checkpoint SHA-256：
  `83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae`；
- 合并表征 SHA-256：
  `910c188914d2e57e86af390857063ff96aee3c97da97509904a5c6e539497ebf`；
- 表征值 hash：`12c58dd4da4fe376`；
- 100 条分层预检全部通过；
- 三臂 × 五 seed × 500 epochs 共 15 个正式训练全部完成，最终 checkpoint 与表征均通过
  合并审计；没有按疾病结果挑 epoch。

## 冻结判据

主比较仍是 matched 上 `correct − within-label` 的配对 `Delta(-NLL)`；配对
`DeltaAUROC`、matched-long 和 L2-normalized 表征是预声明的辅助／敏感性分析。方向与 AST
一致才支持跨 backbone 稳健；不一致必须作为边界条件报告，不能跨 backbone 平均。

## 正式结果

### 绝对结果（raw post-ReLU；AUROC / NLL）

| arm | Standard | matched | matched-long |
|---|---:|---:|---:|
| raw OPERA-CT | 0.7205 / 0.5879 | 0.5772 / 0.7985 | 0.5505 / 0.8354 |
| correct | 0.6814 / 0.6094 | 0.5546 / 0.7826 | 0.5403 / 0.8036 |
| within-label | 0.6658 / 0.6154 | 0.5567 / 0.7650 | 0.5433 / 0.7854 |
| global | 0.6161 / 0.6302 | 0.5339 / 0.7496 | 0.5240 / 0.7577 |

### 主配对比较：correct − within-label

| evaluation | Δ(−NLL) [95% CI] | ΔAUROC [95% CI] |
|---|---:|---:|
| Standard | +0.0060 [+0.0007,+0.0116] | +0.0156 [+0.0082,+0.0230] |
| **matched** | **−0.0176 [−0.0335,−0.0010]** | −0.0020 [−0.0191,+0.0168] |
| matched-long | −0.0182 [−0.0328,−0.0033] | −0.0030 [−0.0153,+0.0094] |

matched 与 matched-long 的 Δ(−NLL) 在五个 seed 中全部为负。L2-normalized 敏感性保留
方向：matched 为 `−0.0130 [−0.0293,+0.0035]`（五个 seed 全负但区间覆盖 0），
matched-long 为 `−0.0150 [−0.0291,−0.0007]`。

## 与 AST 的关系

这次结果支持**主方向跨 backbone 稳健**：

- 两个编码器都在 Standard 上出现 correct 相对 within-label 的正向 AUROC 增量；
- 两个编码器都在 matched 上出现 correct 相对 within-label 的负向 source-calibrated
  Δ(−NLL)，且五个 seed 全负；
- 两个编码器的 matched AUROC 差异都很小且区间覆盖 0；
- matched-long 重复负向 NLL。

OPERA-CT 还给出一条边界信息：它的 raw 表征在 matched 上比 AST 的 raw 表征排序更高
（0.5772 对 0.5380），但 correct alignment 仍没有把正确逐人配对转成可迁移的增量；相对 raw
OPERA-CT，correct 的 matched AUROC 反而低 `−0.0225 [−0.0373,−0.0067]`。因此主结果不是
“AST 太弱才失败”，但仍只适用于当前 Stage-1 projector 与线性 readout。

## 冻结文件

- raw 结果 SHA-256：
  `7b6504e052ab67f9c8a9c3a22346c971b76c9b67e35b41840f3bbd6852d8609d`；
- normalized 结果 SHA-256：
  `59bc717b47cca0e61a7823e99249fc9f08dfc5f0937c361f0ef3d49c466cc094`；
- raw 逐参与者逐 seed 预测 SHA-256：
  `8edc7419423eef43a479a75d9428aefd9203b33927f28e03c8614dc9da90ccb2`；
- normalized 逐参与者逐 seed 预测 SHA-256：
  `3cb7d36f3cf9000532422d240ede11915aa26f33fad5fa88f8ea890fa74449a8`。

最准确的一句话是：

> 换成面向呼吸音预训练的 OPERA-CT 后，正确 metadata 配对仍然在源域学到对应关系，
> 但其相对同标签打乱的优势没有迁移到 matched 疾病预测；主校准指标反而显著变差。
