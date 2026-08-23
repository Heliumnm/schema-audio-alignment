# Metadata alignment 正式实验结果（中文）

> **后续更新（2026-08-23）：** 本文记录首轮 AST 正式结果，数字仍保留；Route-A 后续已补齐
> unique-profile retrieval、OPERA-CT、raw-preserving fusion、probability transport、固定 MLP
> 和 synthetic gate。最终综合判读见 `ICASSP_EXECUTION_STATUS_ZH.md`，不要只引用本文的
> source-calibrated NLL 作为“疾病信息受损”证据。

## 一句话结论

在 UKCOVID 的源数据分布里，正确 metadata 对齐比乱配对齐更容易学习，也能提高
COVID 排序；但到了协变量平衡的 matched 人群，**正确的个人 metadata 配对没有产生
可迁移的个人疾病信息**。事先指定的主指标反而显著变差。正式 probes 进一步表明，
正确配对相对同标签打乱主要保留了个人级性别信息和较弱的年龄信息，但没有稳定增加
症状或招募来源信息；这些个人属性对应也没有转化为 matched 人群中的疾病增益。

本结果属于探索性结果：官方测试集曾参与项目早期的协议设计，任何正向主张都仍需
新的未触碰数据确认。

## 正式设计

冻结 AST 音频编码器与 Phi-2 文本编码器，只训练 RespiraMFM 风格的音频投影层。
正式训练为三个 arm、五个 seed、每组 500 epochs：

- `correct`：音频配本人的结构化 metadata；
- `within_label`：在相同 COVID 标签内打乱 metadata，保留标签级共现、破坏个人对应；
- `global`：全局打乱 metadata；
- `raw_ast`：不做对齐的冻结参照，不重新训练。

训练完整覆盖 Standard train 的 20,714 人；每个 epoch 为 323 个 64 人 batch 加最后
一个 42 人 batch。15/15 个最终 checkpoint 和 15/15 份 72,458 人全队列表征均完成。

主表示是 projector 的 raw post-ReLU 输出；L2-normalized 输出是预先声明的敏感性分析。
每个 projector seed 独立选择下游 logistic head 的 C，并在完整 Standard validation 上
独立拟合 Platt 校准器。主比较为 matched 上 `correct − within_label` 的配对
Δ(−NLL)，置信区间同时重采样参与者和 seed。

## 主结果：raw projector representation

### 各 arm 的绝对表现

| test | raw AST AUROC / NLL | correct | within-label | global |
|---|---:|---:|---:|---:|
| Standard | 0.708 / 0.604 | 0.649 / 0.628 | 0.635 / 0.628 | 0.571 / 0.640 |
| matched | 0.538 / 0.857 | 0.529 / 0.795 | 0.523 / 0.770 | 0.510 / 0.739 |
| matched-long | 0.548 / 0.850 | 0.532 / 0.794 | 0.530 / 0.773 | 0.524 / 0.737 |

### 事先指定的主比较

| test | correct − within-label Δ(−NLL) | 95% CI | ΔAUROC | 95% CI |
|---|---:|---:|---:|---:|
| Standard | +0.0008 | [−0.0047, +0.0062] | +0.0140 | [+0.0040, +0.0235] |
| **matched** | **−0.0249** | **[−0.0443, −0.0069]** | +0.0062 | [−0.0144, +0.0279] |
| matched-long | −0.0214 | [−0.0330, −0.0101] | +0.0013 | [−0.0121, +0.0155] |

matched 上五个 seed 的 Δ(−NLL) 均为负：−0.0260、−0.0513、−0.0204、−0.0092、
−0.0176。该结果不是由单个 seed 拉动。

## normalized 敏感性分析

normalized matched 上 `correct − within_label` 的 Δ(−NLL) 为 −0.0179，
95% CI [−0.0371, +0.0011]；方向相同，但 CI 轻微覆盖 0。matched-long 为 −0.0204，
95% CI [−0.0328, −0.0082]。因此主结果的方向不依赖于是否对 projector 输出做
L2 normalization，但 matched 上的显著性只在预注册的 raw 主表示成立。

## 应该怎样解释

1. **正确个人配对在源域有一点作用，但没有迁移。** Standard 上 correct 相对
   within-label 的 AUROC 增加 0.0140；到了 matched，只剩 +0.0062 且 CI 覆盖 0，
   主 NLL 指标则显著变差。
2. **标签级共现主要是源域信息。** within-label 相对 global 在 Standard 上明显更好；
   到 matched 后 AUROC 差异不确定，NLL 反而显著更差。
3. **不能把 correct 相对 raw AST 的 NLL 改善写成 metadata 对齐成功。** matched 上
   correct 的 NLL 确实低于 raw AST，但 global shuffle 的 NLL 更低。这个改善更像
   projector/正则化降低了源域模型的过度置信，而不是正确语义对应带来了疾病信息。
4. **当前结果不支持马上做 schema 对自然语言的大实验。** 在同样信息内容下比较两种
   文本形式，只有在“正确 metadata 对应本身能产生可测且可迁移的增量”时才有清晰意义；
   这道前提在当前数据上没有成立。

## 正式 probes：对齐究竟保留了什么

正式 probes 已全部完成。下面均为 matched 人群上的辅助属性 AUROC，使用与疾病评测
相同的下游选择协议。`correct − within-label` 隔离个人级对应，
`within-label − global` 描述保留疾病标签共现后的增量，`correct − raw AST` 则比较
投影后表示与未经对齐的冻结 AST。probe 只描述线性可解码信息，不证明疾病分类器实际
使用了该属性。

### Raw projector 表示的绝对 AUROC

| probe | raw AST | correct | within-label | global |
|---|---:|---:|---:|---:|
| 招募来源 | 0.600 | 0.576 | 0.562 | 0.526 |
| 性别 | 0.871 | 0.812 | 0.620 | 0.607 |
| 年龄 ≥65 | 0.742 | 0.657 | 0.611 | 0.563 |
| 是否咳嗽 | 0.728 | 0.653 | 0.643 | 0.563 |
| 无症状 | 0.711 | 0.656 | 0.640 | 0.551 |

### Raw projector 表示的配对比较

| probe | correct − within-label | within-label − global | correct − raw AST |
|---|---:|---:|---:|
| 招募来源 | +0.0138 [−0.0205, +0.0473] | +0.0357 [−0.0049, +0.0743] | −0.0241 [−0.0519, +0.0035] |
| **性别** | **+0.1912 [+0.1677, +0.2156]** | +0.0134 [−0.0117, +0.0392] | **−0.0590 [−0.0715, −0.0456]** |
| **年龄 ≥65** | **+0.0465 [+0.0069, +0.0854]** | **+0.0473 [+0.0079, +0.0869]** | **−0.0851 [−0.1154, −0.0533]** |
| 是否咳嗽 | +0.0098 [−0.0114, +0.0309] | **+0.0802 [+0.0561, +0.1037]** | **−0.0755 [−0.0946, −0.0563]** |
| 无症状 | +0.0151 [−0.0096, +0.0407] | **+0.0895 [+0.0613, +0.1184]** | **−0.0553 [−0.0760, −0.0345]** |

括号为 participant × seed 分层 bootstrap 的 95% CI。个人级正确配对相对
within-label 的稳定增量集中在性别；年龄的 raw 主表示 CI 也排除 0，但效应更小。
咳嗽、无症状和招募来源在 matched 上均没有可检出的个人级增量。相反，症状的
`within-label − global` 明显为正，说明其可解码性主要可由标签级／队列级共现保留，
不能归因于正确的逐患者 metadata 对应。

### L2-normalized 敏感性分析

| probe | raw correct − within-label | normalized correct − within-label |
|---|---:|---:|
| 招募来源 | +0.0138 [−0.0205, +0.0473] | +0.0182 [−0.0144, +0.0499] |
| **性别** | **+0.1912 [+0.1677, +0.2156]** | **+0.1980 [+0.1756, +0.2216]** |
| 年龄 ≥65 | +0.0465 [+0.0069, +0.0854] | +0.0335 [−0.0041, +0.0734] |
| 是否咳嗽 | +0.0098 [−0.0114, +0.0309] | +0.0092 [−0.0112, +0.0303] |
| 无症状 | +0.0151 [−0.0096, +0.0407] | +0.0174 [−0.0070, +0.0429] |

归一化不改变主要形状：性别增量稳定且很大；症状和招募来源仍不确定；年龄仍同向，
但 normalized matched CI 覆盖 0。matched-long 上性别与年龄在 raw 和 normalized
表示中均同向且 CI 排除 0；其他属性没有跨表示、跨 matched 集合的一致个人级增量。

### Probes 的准确结论边界

1. **能说：**与架构相同但个人配对被破坏的 within-label projector 相比，正确
   metadata 配对选择性地保留了性别以及部分年龄对应信息。
2. **不能说：**metadata alignment 让表示比 raw AST 包含更多人口学或症状信息。
   raw AST 对五个 probe 的绝对 AUROC 均高于 correct；除招募来源外，
   `correct − raw AST` 的 CI 也都明确为负。projector 总体压缩了 raw AST 中原有的
   可解码信息，只是正确配对改变了相对哪些信息被保留下来。
3. **不能说：**疾病分类器依赖了性别或年龄。probe 测量的是表示中可以被线性读出的
   信息，不是疾病 head 的因果使用证据。
4. **与疾病主结果合起来能说：**模型学到了真实的个人 metadata 对应关系，尤其是
   性别对应；但这种对应没有产生可迁移的 matched 疾病证据。这比笼统说“对齐完全
   没学到东西”更准确，也比说“对齐写入了更多患者属性”更克制。
