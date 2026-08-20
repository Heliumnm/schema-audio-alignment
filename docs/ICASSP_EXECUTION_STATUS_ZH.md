# ICASSP 执行状态（2026-08-20）

## 论文现在讲什么

这是一篇临床音频 metadata alignment 的受控审计，不是“提出更强模型”：

> 当训练标签几乎等于招募来源时，正确的音频—metadata 配对可以学到真实的患者对应，
> 但这种对应是否会变成协变量平衡人群里可迁移的疾病证据？

关键控制是 `within-label shuffle`：只在相同 COVID 标签内打乱 metadata，保留标签级共现、
破坏逐患者对应。`correct − within-label` 因此隔离逐患者正确配对本身带来的东西。

## 已完成的主结果

### 1. AST-6L 配对控制

- 招募来源预测 COVID：Standard train AUROC `0.9966`，matched 为 `0.5000`；
- Standard 上 `correct − within-label` 的 ΔAUROC 为
  `+0.0140 [+0.0040,+0.0235]`，说明正确配对确实学到了个人对应；
- matched 主结果 Δ(−NLL) 为 `−0.0249 [−0.0443,−0.0069]`，五个 seed 全为负；
- matched ΔAUROC 为 `+0.0062 [−0.0144,+0.0279]`，没有可检出的排序收益；
- matched-long 的 NLL 方向一致；
- probe 显示 correct 相对 within-label 明显保留性别、较弱保留年龄。

准确结论不是“alignment 什么也没学”，而是：**它学到了患者对应，却没有改善 matched
人群中的校准疾病预测。**

### 2. Day 1：matched direct-fusion attribution control

这一步已完成并冻结。在 matched 的 1,814 人上：

- `metadata + raw AST − metadata only`：ΔAUROC
  `+0.0028 [−0.0019,+0.0071]`；Δ(−NLL) `−0.0034 [−0.0194,+0.0126]`；
- `metadata + correct − metadata + raw AST`：ΔAUROC
  `−0.0040 [−0.0093,+0.0015]`；Δ(−NLL) `−0.0077 [−0.0301,+0.0157]`。

两项区间都覆盖 0。因此 matched 上缺少增量不能全部归咎于 alignment：原始 AST 在同一
schema metadata 之上本来就没有可检出的稳定疾病增量，correct alignment 也没有救回来。

### 3. Coswara 外部门槛

12.98-GB Zenodo v1.0 已完成官方 MD5／ZIP 结构核验；2,746 条 `cough-heavy.wav` 全部
提取并解码，2,646 条通过客观 QC。冻结匹配最多 112 对；在 100 对的预注册下限上，最大
类别 SMD 仍为 `0.140 > 0.12`，所以正式门槛为 **NO-GO**。

没有生成 Coswara 表征、分类器或模型分数；没有放宽阈值、换录音或改匹配字段。论文只把
它写成数据可行性结果，不冒充外部确认。

## Day 2–4：第二音频编码器 OPERA-CT

已经完成：

- 100 条分层技术预检全部通过；
- `72,458 / 72,458` 名参与者成功提取为 768 维表征，全部有限、非零、顺序逐位一致；
- 三个配对臂 × 五个 seed × 500 epochs 共 15 个正式训练全部完成；
- checkpoint、表征、配对和输入 hash 全部通过合并审计；
- 没有按疾病结果选择 epoch。

raw 与 normalized 两套冻结 disease evaluator 均已完成。主 raw 结果：

- Standard：`correct − within-label` 的 ΔAUROC
  `+0.0156 [+0.0082,+0.0230]`，Δ(−NLL) `+0.0060 [+0.0007,+0.0116]`；
- matched：Δ(−NLL) `−0.0176 [−0.0335,−0.0010]`，五个 seed 全负；ΔAUROC
  `−0.0020 [−0.0191,+0.0168]`；
- matched-long：Δ(−NLL) `−0.0182 [−0.0328,−0.0033]`；
- normalized 敏感性保持负向，matched-long 的区间仍排除 0。

因此第二编码器支持 AST 的主方向：正确配对在源域学到对应关系，但这个优势没有迁移为
matched 疾病收益，source-calibrated NLL 反而更差。OPERA 的 raw matched AUROC 为 0.5772，
高于 AST 的 0.5380，所以不能把结果简单归因于“AST 太弱”。

## 当前收尾顺序

1. OPERA 绝对结果、配对区间和逐参与者预测 hash 已冻结；
2. 中英文结果文档与 ICASSP 草稿正在更新；
3. 重新构建四页 PDF，检查图表、引用、溢出和页数；
4. 提交并推送 GitHub。

## 仍然不能说的话

- 不说 metadata alignment 普遍无效；
- 不说 AST／OPERA 完全没有疾病信息；
- 不把 UKCOVID discovery 叫独立确认；
- 不把 Coswara NO-GO 写成模型失败；
- 不把一个线性 readout 的 null 推广到所有非线性分类器；
- 不把 AUROC 区间覆盖 0 写成等效性证明。
