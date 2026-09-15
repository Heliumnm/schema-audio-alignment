# Coswara 三骨干 post-hoc 外部压力测试结果

日期：2026-09-15
冻结执行：`locked_panel_v3_20260914`
协议地位：原预注册贪心数据门仍为 `NO_GO`；本结果来自在读取任何模型分数前另行冻结的
全局约束 secondary cohort，因此只能作为 post-hoc external stress test，不能冒充 untouched
external confirmation。

## 一句话结论

> AST-6L、OPERA-CT 与 HeAR 都学到了正确音频--metadata 的逐参与者对应，并稳定增强了性别
> 信息的可解码性；但是，在冻结的100对协变量平衡 COVID target 上，Correct 相对
> Within-label 没有带来可检出的疾病迁移收益。

这再次支持论文的核心评价歧义：**correspondence learning 成功，不等于 transferable disease
evidence 已经增加。** 它不证明 metadata alignment 普遍无效，也不证明效应严格为0。

## 1. 数据门与执行边界

- 可用 cohort：1,701 名 participant；
- train：1,041；validation：217；source-test：243；
- matched target：200人，即100对正负样本；
- matched target 年龄 SMD 为0，最大分类 SMD 为0.1034，最大层级比例差为0.07；
- train／validation／source-test／matched participant 完全隔离；
- 三个 backbone 均完成 Correct／Within-label／`W_{y,s}`／Global × 5 seeds × 500 epochs；
- target 在模型训练、source-validation readout 与校准冻结后才用于正式打分。

原贪心 gate 在100对下得到最大 SMD 0.140，超过0.12门槛，因此该结论继续保留。后来的全局
约束 cohort 是透明版本化的 secondary 分析，不用于改写原 gate。

## 2. Alignment 是否建立了 correspondence？

validation 上有217名 query participant、186个唯一 metadata profile。主比较为
`Correct − Within-label`：后者保留相同 COVID 标签，但破坏逐参与者配对。

| backbone | Correct MRR | Within MRR | `W_{y,s}` MRR | Correct−Within，95% CI | Correct−`W_{y,s}`，95% CI |
|---|---:|---:|---:|---:|---:|
| AST-6L | 0.0521 | 0.0401 | 0.0407 | +0.0120 [−0.0044, 0.0280] | +0.0114 [−0.0032, 0.0262] |
| OPERA-CT | 0.0552 | 0.0357 | 0.0397 | **+0.0196 [0.0029, 0.0387]** | +0.0155 [−0.0053, 0.0377] |
| HeAR | 0.0609 | 0.0337 | 0.0484 | **+0.0272 [0.0102, 0.0451]** | +0.0125 [−0.0063, 0.0315] |

三个点估计均为正，其中 OPERA-CT 与 HeAR 的 Correct−Within 区间排除 0；AST 区间跨 0。
整体仍表明配对干预产生了可测的 profile correspondence，但不能把每个 backbone 都写成
独立确证。

## 3. Correspondence 是否转化为 matched COVID transfer？

| backbone | Raw | Correct | Within | `W_{y,s}` | `C-W_{y,s}` ΔAUROC，95% CI | `W_{y,s}-W`，95% CI |
|---|---:|---:|---:|---:|---:|---:|
| AST-6L | 0.6772 | 0.6643 | 0.6734 | 0.6510 | +0.0132 [−0.0389, 0.0611] | −0.0224 [−0.0734, 0.0283] |
| OPERA-CT | 0.6349 | 0.6020 | 0.6122 | 0.6172 | −0.0152 [−0.0727, 0.0326] | +0.0050 [−0.0393, 0.0510] |
| HeAR | 0.6155 | 0.5770 | 0.5917 | 0.5847 | −0.0076 [−0.0579, 0.0507] | −0.0071 [−0.0647, 0.0591] |

对应 `Correct−W_{y,s} Δ(−NLL)` 为：

- AST：+0.0155 [−0.0253, 0.0514]；
- OPERA：−0.0167 [−0.0545, 0.0137]；
- HeAR：−0.0180 [−0.0600, 0.0301]。

三个 backbone 的 AUROC 与 NLL 区间都跨0，也没有同时完整落入预注册等价区间。因此准确判定
是 `correspondence_gain_but_matched_transfer_inconclusive`：没有正向疾病迁移证据，但也不能
声称已经证明严格等价或零效应。

source-test 的 Correct−Within ΔAUROC 分别为 AST +0.0240、OPERA +0.0536、HeAR +0.0417，
三个区间同样跨0。源域点估计偏正不能替代 matched target 上的证据。

## 4. Correct pairing 更稳定地保留了什么？

matched target 上的性别 probe：

| backbone | Correct−Within ΔAUROC，95% CI |
|---|---:|
| AST-6L | **+0.1118 [0.0335, 0.2080]** |
| OPERA-CT | **+0.1072 [0.0405, 0.1844]** |
| HeAR | **+0.0469 [0.0107, 0.0943]** |

三个 backbone 均显著为正，而 matched COVID 增量接近0且区间跨0。因此最稳健的通道不是疾病
标签，而是患者人口学对应信息。Age≥45 的方向在三者中均为正，但区间均跨0；其他症状、国家
与录音属性 probes 多数依赖 backbone。Probe 只说明信息可解码，不证明疾病分类器因果使用它。

## 5. 对论文的作用

Coswara 与 UKCOVID、CODA TB、Cambridge 的数据来源和 cohort 构造不同，却重复出现：

```text
Correct pairing
    -> profile correspondence 增加
    -> patient sex information 增加
    -> matched disease transfer 没有建立
```

这提高了 audit framework 的跨数据集一致性。不过本 cohort 是原数据门失败后的 secondary
全局约束构造，所以论文中应称为 **post-hoc external stress test**，不能与 untouched external
confirmation 等价。

## 6. 可审计文件

- `results/coswara_formal_results_ast.json`；
- `results/coswara_formal_results_opera_ct.json`；
- `results/coswara_formal_results_hear.json`。

SHA-256：

| 文件 | SHA-256 |
|---|---|
| AST | `a91bc912c4d109e7ade4ac5c74658a54bcf784505ab7ec3ef3cad89ca05c5843` |
| OPERA-CT | `2e47036ed680884f71514d5422d4d06d6ee4b5496f1bbe032857091c29ed9c3f` |
| HeAR | `36132476cc3573920f54c9eaea912fde30a89ff3cb237abe1dc48619c620b4a3` |

音频、逐参与者 metadata、private participant/pair manifest、embedding、prediction 与
checkpoint 保留在受控服务器，不进入 GitHub；公开仓库仅同步不含参与者标识的 aggregate
result 与训练哈希 manifest。
