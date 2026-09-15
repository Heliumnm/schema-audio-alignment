# UKCOVID E1--E3 定向诊断：最终结果

日期：2026-09-13
性质：**主结果读取后冻结并执行的定向诊断；HeAR 部分是额外的事后第三骨干扩展。**

冻结协议：

- AST-6L／OPERA-CT：`docs/PAIRING_AUDIT_SUPPLEMENTARY_EXPERIMENTS_PREREG_ZH.md`
- HeAR：`docs/PAIRING_AUDIT_HEAR_FOLLOWUP_FREEZE_ZH.md`

## 一句话结论

加入“同疾病标签且同性别打乱”的新对照 (W_{y,s}) 后，三个 backbone 原有的
Correct−Within 性别探针对比均被大幅削弱，完整区间覆盖 0；同时，profile retrieval 仍留有
residual exact-pairing signal，AST 和 HeAR 的区间仍排除 0。因此只能说性别探针对比对配对规则
是否保留记录性别一致性高度敏感，不能把它扩大成“性别解释了全部配对收益”。无论使用
source-only readout，还是用 matched-long 标签训练的 target-assisted readout，都没有建立跨
backbone 一致的 matched COVID 收益，但个别区间仍允许有意义的正向效应。

因此，新增诊断**收窄并加强**了原结论：alignment 确实学习了配对结构；原来最大的性别探针
差异对记录性别一致性敏感；控制这一条件后仍存在部分 profile correspondence，但没有形成可重复
的疾病迁移。区间跨零不是等效性证明，也不排除单个设置存在正向收益。

## 1. E1：只改变 retrieval 候选库

E1 不重新训练 projector，只在 Standard validation 的 5,179 个 query、1,871 个唯一 profile 上
限制候选资格。下表为同一候选条件内的 Correct−Within macro-profile MRR：

| backbone | unrestricted | same sex | same label | same label + sex |
|---|---:|---:|---:|---:|
| AST-6L | +0.00318 [0.00161, 0.00492] | +0.00140 [−0.00065, 0.00350] | +0.00470 [0.00283, 0.00666] | +0.00245 [−0.00001, 0.00499] |
| OPERA-CT | +0.00323 [0.00155, 0.00500] | +0.00060 [−0.00143, 0.00256] | +0.00459 [0.00256, 0.00670] | +0.00094 [−0.00149, 0.00329] |
| HeAR（post-hoc） | +0.00711 [0.00476, 0.00971] | +0.00356 [0.00096, 0.00638] | +0.00923 [0.00623, 0.01240] | +0.00451 [0.00115, 0.00813] |

AST／OPERA 在同性别候选库中明显缩小并跨零；HeAR 仍为正。因此，候选库中的性别结构能解释
一部分 retrieval 优势，但不能解释 HeAR 的全部结果。不同候选库的随机基线和规模不同，不能把
缩小比例解释为“性别贡献百分比”。

## 2. E2：训练新的性别保持打乱对照

新臂定义为：

\[
a_i \leftrightarrow m_j,\qquad y_i=y_j,\quad s_i=s_j,\quad i\ne j,
\]

记为 (W_{y,s})。训练参与者仍为 20,714 人；没有删除单例、没有自配对回退，也没有跨性别
配对。每个 backbone 使用 seeds 0--4、500 epochs，并逐 seed 复用原 Within-label 的初始化和
batch-order hash。

该规则保证换到不同参与者，但不保证换到不同语义 profile。逐 seed 核查显示，原 W 中重新配对后
仍命中相同 schema text 的比例为 3.03%--3.33%，(W_{y,s}) 为 6.35%--6.61%。更细分层提高了
重复 profile 碰撞率，因此图示和文字均应写“different participant”，不能写“no exact profile”。
聚合审计保存在 `results/pairing_followup/e2_pairing_collision_audit.json`。

### 2.1 性别探针对比被大幅削弱

| backbone | (C-W_{y,s}) sex ΔAUROC | (W_{y,s}-W) sex ΔAUROC |
|---|---:|---:|
| AST-6L | −0.0004 [−0.0153, 0.0147] | +0.1916 [0.1683, 0.2142] |
| OPERA-CT | +0.0067 [−0.0054, 0.0192] | +0.1058 [0.0896, 0.1222] |
| HeAR（post-hoc） | +0.0064 [−0.0011, 0.0136] | +0.0797 [0.0667, 0.0928] |

三个 (C-W_{y,s}) 区间都覆盖 0，而三个 (W_{y,s}-W) 都明确为正。这直接支持：原有
Correct−Within sex probe 差异对 correct pairing 是否保留记录性别一致性高度敏感。它不是等效性
分析，也不能证明疾病分类器因果使用了性别；按标签与性别分层还会一并保留与性别相关的其他
metadata 结构。

### 2.2 控制性别后仍有小的 profile correspondence

| backbone | 原 (C-W) MRR | (C-W_{y,s}) MRR | (W_{y,s}-W) MRR |
|---|---:|---:|---:|
| AST-6L | +0.00318 [0.00160, 0.00492] | +0.00164 [0.00001, 0.00335] | +0.00154 [−0.00004, 0.00322] |
| OPERA-CT | +0.00323 [0.00160, 0.00495] | +0.00099 [−0.00096, 0.00300] | +0.00224 [0.00020, 0.00417] |
| HeAR（post-hoc） | +0.00711 [0.00475, 0.00965] | +0.00394 [0.00175, 0.00637] | +0.00318 [0.00141, 0.00483] |

残余 exact-pairing MRR 在 AST 上刚好高于零，在 OPERA 上跨零，在 HeAR 上明确为正。准确表述是：
性别一致性不是全部 correspondence，但去掉它后，剩余效应很小且不具三个 backbone 的一致明确性。

### 2.3 症状与疾病

HeAR 的 (C-W_{y,s}) 对 cough-any 为 +0.0179 [0.0042, 0.0327]，对 no-symptoms 为
+0.0181 [0.0005, 0.0360]；AST／OPERA 的相应区间覆盖 0。因此 HeAR 仍保留一部分逐 profile
症状信息，但这不是跨 backbone 的统一效应。

source-only matched COVID 的完整结果为：

| backbone | (W_{y,s}) absolute AUROC | (C-W_{y,s}) ΔAUROC | (W_{y,s}-W) ΔAUROC |
|---|---:|---:|---:|
| AST-6L | 0.529 [0.504, 0.554] | −0.0003 [−0.0228, 0.0241] | +0.0065 [−0.0197, 0.0313] |
| OPERA-CT | 0.554 [0.529, 0.577] | +0.0009 [−0.0169, 0.0196] | −0.0029 [−0.0219, 0.0164] |
| HeAR（post-hoc） | 0.532 [0.507, 0.557] | +0.0130 [−0.0023, 0.0287] | −0.0102 [−0.0256, 0.0053] |

六个配对 AUROC 区间均覆盖 0；HeAR 的 (C-W_{y,s}) 五个 seed 点估计同向，但预先冻结的配对
区间仍未建立正向收益。对应 Δ(−NLL) 也都覆盖 0。

## 3. E3：target-assisted readout

E3 只在 matched-long 上训练、选正则并校准固定 logistic readout，再一次性应用到 participant-
disjoint matched。它使用目标分布标签，因此只回答“source readout 是否掩盖了表示中的信号”，
不能与 source-only 主结果混为一谈。

| backbone | (C-W) ΔAUROC | (C-W_{y,s}) ΔAUROC | (C-Raw) ΔAUROC |
|---|---:|---:|---:|
| AST-6L | −0.0082 [−0.0337, 0.0177] | −0.0182 [−0.0501, 0.0122] | **−0.0504 [−0.0843, −0.0182]** |
| OPERA-CT | +0.0287 [−0.0017, 0.0626] | +0.0019 [−0.0210, 0.0254] | −0.0134 [−0.0419, 0.0147] |
| HeAR（post-hoc） | +0.0047 [−0.0278, 0.0408] | +0.0114 [−0.0117, 0.0333] | +0.0071 [−0.0241, 0.0376] |

所有 Correct−Within、Correct−(W_{y,s}) 和对应 Δ(−NLL) 区间都覆盖 0。Correct−Raw 也没有
正向证据；AST 反而明确低于 raw。目标辅助读出没有建立跨 backbone 一致的 alignment 优势；但
OPERA 的 +0.0287 [−0.0017, 0.0626] 仍允许正向效应。source-only 与 target-assisted 估计之间
没有直接做差，因此不能声称已经排除所有 readout 限制，也不能证明表示中完全没有疾病信息。

## 4. 最终科学判读

1. **Alignment 发生了。** 三个 backbone 的原始 Correct−Within retrieval 均明确为正。
2. **原来最强的 sex probe 对配对规则敏感。** 保留记录性别一致性会大幅削弱 (C-W_{y,s})，
   但这不是因果分解或等效性证明。
3. **性别不是全部 pairing signal。** HeAR 和边界上的 AST 仍有 residual retrieval，HeAR 还保留
   部分症状 correspondence。
4. **Correspondence 没有变成稳定疾病迁移。** source-only 与 target-assisted 两套 readout 都未
   建立跨 backbone 的 matched COVID 优势，也没有稳定胜过 raw audio；不确定性仍允许单个设置
   存在有意义的正效应。
5. **不能声称因果分解或真实零效应。** (C-W_{y,s}) 不是严格因果估计，区间覆盖零也不是等效性
   证明。

最简洁的论文结论是：

> Correct pairing 的确让音频表示更贴近对应的患者资料；原来最大的性别探针对比在保留记录
> 性别一致性的条件化对照下被大幅削弱，但仍有部分 profile correspondence。source-only 与
> target-assisted 读出都没有建立跨 backbone 一致的协变量平衡 COVID 收益。

## 5. 置信区间的实际重采样单位

- macro-profile retrieval：以唯一 metadata profile 为单位，并与 seed 轴联合重采样；同一
  profile 的重复 participant query 先在 profile 内平均。
- probe、source-only disease 和 target-assisted disease：以 participant 为单位，并与 seed 轴
  联合重采样；同一 participant、同一 seed 上的 arm 预测始终配对。
- UKCOVID 的实现没有把推定的病例--对照 pair ID 当作 bootstrap block；正文已明确写为
  participant resampling，而不是 matched-pair resampling。

## 6. 可复核归档

公开仓库只保存不含 participant identifier 的聚合 JSON、配置与 manifest：

- `results/pairing_followup/e1_{ast,opera_ct,hear}/`
- `results/pairing_followup/e2_alignment_{ast,opera_ct,hear}/manifest.json`
- `results/pairing_followup/e2_retrieval_{ast,opera_ct,hear}/`
- `results/pairing_followup/e2_eval_{ast,opera_ct,hear}/metrics.json`
- `results/pairing_followup/e3_target_{ast,opera_ct,hear}/metrics.json`
- `results/pairing_followup/e2_pairing_collision_audit.json`

HeAR 最终聚合文件的服务器／本地 SHA-256 逐项一致：

| 文件 | SHA-256 |
|---|---|
| E1 metrics | `0dbe017ff69aa446e4d38492e0fe1b762e7dbc0bc5d9d459aab21adfefb9cc5d` |
| E2 manifest | `123a0d34f8c034f6fc83b295cec6f3712bf4407d128e40a2e0ce525701ac1591` |
| E2 retrieval metrics | `363b87da798c5aeebc306ae064927929ffd6c18f3182aee7e758b81b49854575` |
| E2 eval metrics | `859b0eb31af066827dbc5bc457270e2090823feccc714d4c67e634756d27c5d3` |
| E3 metrics | `fda41ed80975668ed93501b7a941f5e5b2ee0c90f43365742473fe8e6c0939b0` |

HeAR E2 的五个 seed 均完成 500 epochs；manifest 明确记录
`matched_or_test_labels_read=false`。含逐参与者预测、query ranks、projector 和 embedding 的大型文件
只留在受控服务器，不进入公开仓库。
