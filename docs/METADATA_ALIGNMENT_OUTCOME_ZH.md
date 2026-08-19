# Metadata alignment 正式实验结果（中文）

## 一句话结论

在 UKCOVID 的源数据分布里，正确 metadata 对齐比乱配对齐更容易学习，也能提高
COVID 排序；但到了协变量平衡的 matched 人群，**正确的个人 metadata 配对没有产生
可迁移的个人疾病信息**。事先指定的主指标反而显著变差。

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

## 仍在运行

性别、年龄、症状和 recruitment source 的正式 probes 已在服务器后台运行。它们用于
解释 alignment 把哪些变量写入表示，不改变上述预注册疾病主结果；“可解码”也不等于
疾病分类器实际使用了该变量。

