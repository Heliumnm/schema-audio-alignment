# Coswara 三骨干 post-hoc 外部压力测试结果

日期：2026-09-04  
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
- 三个 backbone 均完成 Correct／Within-label／Global × 5 seeds × 500 epochs；
- target 在模型训练、source-validation readout 与校准冻结后才用于正式打分。

原贪心 gate 在100对下得到最大 SMD 0.140，超过0.12门槛，因此该结论继续保留。后来的全局
约束 cohort 是透明版本化的 secondary 分析，不用于改写原 gate。

## 2. Alignment 是否建立了 correspondence？

validation 上有217名 query participant、186个唯一 metadata profile。主比较为
`Correct − Within-label`：后者保留相同 COVID 标签，但破坏逐参与者配对。

| backbone | Correct MRR | Within MRR | Global MRR | Correct−Within，95% CI |
|---|---:|---:|---:|---:|
| AST-6L | 0.0570 | 0.0387 | 0.0286 | **+0.0183 [0.0028, 0.0365]** |
| OPERA-CT | 0.0615 | 0.0373 | 0.0267 | **+0.0243 [0.0050, 0.0452]** |
| HeAR | 0.0729 | 0.0323 | 0.0316 | **+0.0406 [0.0226, 0.0619]** |

三个区间均排除0，所以不能把后续负向或不确定的 transfer 读成“projector 没学到东西”。
它明确学到了正确配对所携带的 profile correspondence。

## 3. Correspondence 是否转化为 matched COVID transfer？

| backbone | Raw AUROC | Correct | Within | Global | Correct−Within ΔAUROC，95% CI |
|---|---:|---:|---:|---:|---:|
| AST-6L | 0.6772 | 0.6811 | 0.6783 | 0.5971 | +0.0028 [−0.0450, 0.0464] |
| OPERA-CT | 0.6348 | 0.6043 | 0.6159 | 0.5564 | −0.0115 [−0.0750, 0.0444] |
| HeAR | 0.6153 | 0.5927 | 0.5890 | 0.5552 | +0.0037 [−0.0646, 0.0724] |

对应 `Correct−Within Δ(−NLL)` 为：

- AST：+0.0046 [−0.0333, 0.0412]；
- OPERA：−0.0148 [−0.0546, 0.0190]；
- HeAR：−0.0025 [−0.0429, 0.0397]。

三个 backbone 的 AUROC 与 NLL 区间都跨0，也没有同时完整落入预注册等价区间。因此准确判定
是 `correspondence_gain_but_matched_transfer_inconclusive`：没有正向疾病迁移证据，但也不能
声称已经证明严格等价或零效应。

source-test 的 Correct−Within ΔAUROC 分别为 AST +0.0263、OPERA +0.0482、HeAR +0.0294，
三个区间同样跨0。源域点估计偏正不能替代 matched target 上的证据。

## 4. Correct pairing 更稳定地保留了什么？

matched target 上的性别 probe：

| backbone | Correct−Within ΔAUROC，95% CI |
|---|---:|
| AST-6L | **+0.1324 [0.0511, 0.2246]** |
| OPERA-CT | **+0.1075 [0.0469, 0.1699]** |
| HeAR | **+0.0570 [0.0189, 0.0990]** |

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
| AST | `63236138350188a6b05b678ce047db3c05cd9f6155cc0166dc2df8ca09658d61` |
| OPERA-CT | `bea539a8fe0a3da92d97b8f20238094a1e7db9a2c4de877552b0a007af505bf4` |
| HeAR | `d722c1025afa2c55aaa6ae8b142d33f96f0bcfec6a38a6882eb77c99de894997` |

音频、逐参与者 metadata、manifest、embedding、prediction 与 checkpoint 保留在受控服务器，
不进入 GitHub。
