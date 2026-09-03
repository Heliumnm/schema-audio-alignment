# Coswara 全局约束匹配：二次数据门敏感性协议

冻结日期：2026-09-03  
状态：**在读取任何 Coswara 模型表示或预测之前冻结**

## 为什么允许再检查一次

原 Coswara 数据门已经是 match-first，而不是 split-first。它先得到 112 对最大匹配，再沿一条
确定性的贪心路径逐对删除；到最低允许的 100 对时，最大分类 SMD 为 0.1401，超过冻结门槛
0.12，因此正式结论是 `NO-GO`。

这个结果证明“那一条贪心删除路径没有找到合格队列”，但没有证明“所有可能的 100 对队列都
不合格”。因此只允许做一次不读取模型分数的全局可行性复核。它不是重新随机切分，也不允许
放宽原门槛。

## 冻结不变的内容

- 同一份 `cough-heavy.wav` 客观 QC 与 PCM 去重结果；
- 同一 participant eligibility、COVID/test-status 标签定义和 760 名严格候选人；
- 同一 exact matching 字段、country grouping 和 metadata 解释；
- 固定 100 个正负配对；
- 最大绝对分类 SMD `<=0.12`；
- 最大细平衡层级比例差 `<=0.08`；
- participant/PCM 跨 split 为零，开发集规模门不变；
- feasibility 阶段禁止读取 AST、OPERA、HeAR、projector、retrieval 或疾病预测。

## 唯一变化

原实现先做 minimum-cost 最大匹配，再贪心删 12 对。本敏感性分析改用一次全局混合整数可行性
求解，在所有合格 participant 中同时选择正负各 100 人：

- 每个 exact stratum 的正负人数必须完全相等；
- 年龄总和必须完全相等，因此年龄均值差和年龄 SMD 都为零；
- 原 SMD 字段的每个 level 按冻结的 binary-SMD 公式精确限制在 0.12 内；
- 原 fine-balance 字段每个 level 的人数差最多 7/100。该条件比原来的 `<=8/100` 更严格，
  用于避免二进制浮点把数学上的 0.08 记成 0.08000000000000002；
- 选中 participant 后，只在各 exact stratum 内做冻结 cost 的 Hungarian pairing。

## 判决

- 找到 100 对且原 `balance_report` 复算全部通过：记作 `SECONDARY_GO`；随后才允许冻结新的
  participant-level train/validation/source-test，并把模型实验称为 **post-hoc external stress
  test**，不能升级为预注册外部确认。
- 求解器证明上述更严格问题 infeasible：保留原 `NO-GO`。因为年龄与 fine-balance 使用了更强
  条件，不得声称已经证明原容差范围内不存在任何可行集合。
- 超时或数值失败：记作 `INCONCLUSIVE`，不得启动模型。

不论结果如何，不改变原 `docs/COSWARA_DATA_GATE_OUTCOME_ZH.md`；二次结果单独报告。
