# CODA TB 外部敏感性分析 v2：match-first 数据门预注册

冻结日期：2026-09-03  
状态：在读取 v1 聚合数据门结果之后、读取任何 CODA 模型输出之前冻结

## 1. 为什么需要 v2

v1 先以 `label × country × sex` 分层，把 60% participant 分给 development、40% 分给
target candidate，再只在后者中匹配。1,081 名 eligible participant 因而只有约 445 人能够
参与 target matching；初始得到 117 对，沿冻结删减路径保留 100 对后，最大绝对 SMD 为
0.276、最大分类水平比例差为 0.13，未通过 0.12／0.08 平衡门。

v2 只检验一个预先声明的问题：**v1 的 NO-GO 是不是由“先切分、后匹配”限制了 common
support 搜索空间？** 因此改成先在全部 eligible participant 中构造并冻结 matched target，
再对未进入 target 的 participant 划分 development。

这是看过 v1 数据门统计后建立的 model-blind secondary sensitivity analysis，不冒充最初的
untouched confirmatory analysis。v1 结果永久保留，不被 v2 替代。

## 2. 不能改变的部分

以下内容与 v1 完全相同：

- participant 是唯一分析单位；
- primary audio 仅为 clinic-supervised solicited cough；
- TB label、metadata 清洗、音频 QC、PCM 去重和 profile 定义；
- country × sex 精确匹配；
- pair cost 使用 age、height、weight、log cough duration、heart rate、temperature、prior TB、
  hemoptysis、weight loss、recent smoking、fever、night sweats 和 HIV；
- Hungarian maximum-cardinality assignment；
- `coda-match-v1` pair tie-break 和 v1 唯一 greedy trimming path；
- target 为 **100 对** TB+/TB− participant；
- 最大绝对 SMD `<=0.12`；
- 最大分类水平比例差 `<=0.08`；
- development train、validation、source-test 的样本量与类别下限；
- schema/profile diversity、split overlap、PCM overlap 和 model-blind gates；
- 不加载 AST/OPERA，不读取 embedding、prediction、AUROC、NLL 或 calibration。

不得通过更换随机种子、匹配字段、distance weight、删减路径或阈值寻找有利结果。

## 3. 唯一改变：target 先于 development

1. 完成 v1 相同的 linkage、QC、去重和 participant eligibility；
2. 全部 eligible participant 暂时进入 matching candidate pool；
3. 在每个 country × sex stratum 内执行 v1 的 minimum-cost assignment；
4. 沿 v1 唯一 greedy path 删减到恰好 100 对；即使更大的集合提前达到平衡也继续沿同一路径
   到 100 对，以固定 target power 并给 development 保留容量；
5. 这 200 名 participant 冻结为 `matched_target`，永不回流 development；
6. 剩余 participant 以 `label × country × sex` 分层，沿 v1 的 `coda-dev-v1` 哈希顺序划分为
   70% train、15% validation、15% source-test；
7. matched target 不选择 encoder、projector、epoch、classifier、regularisation 或 calibration。

## 4. GO／NO-GO

v1 的全部数据与功效门继续适用。特别是：

- matched target 必须恰好为 100 对；
- country 与 sex 必须 pairwise exact；
- 最大绝对 SMD `<=0.12`；
- 最大分类水平比例差 `<=0.08`；
- train 至少400人且每类至少100人；
- validation 和 source-test 各至少75人且每类至少15人；
- participant 和 decoded PCM 不得跨 split；
- 本阶段不得生成任何模型输出。

全部通过才允许另行冻结 CODA 模型实验。任一项失败即关闭 CODA v2，不建立 v3，也不放宽
门槛。

## 5. 允许的结论

- 若 GO：match-first 能在 CODA training release 内建立一个平衡的 model-blind external
  sensitivity endpoint；正式模型结果仍不等同于官方 CODA challenge validation。
- 若 NO-GO：即使让全部 eligible participant 参与 target selection，冻结规则仍无法同时满足
  100 对和协变量平衡；这仍是数据可行性结论，不是模型或 TB 声学信号的负结果。

