# 从“对齐成功”到“数据不支持修复”：Transfer Repair 的完整中文故事

日期：2026-09-04

## 一句话版本

> 把医疗音频与患者 metadata 对齐，模型确实会学会“这段声音对应什么样的患者”，但这不等于
> 学到了能跨人群使用的疾病声音。我们先用配对控制把这两件事分开，又尝试删除捷径和重新设计
> disease-invariant 对比学习；最终发现，现有单个数据集缺少训练这种修复目标所需的共同支持。
> 所以当前贡献是一个严格的 alignment audit，而不是一个已经成功的 repair 方法。

## 1. 最开始的直觉为什么合理

临床音频数据通常同时有声音和患者资料：年龄、性别、症状、吸烟史、既往疾病等。一个自然的
做法是把音频表示与这些文字做对比学习：正确配对拉近，错误配对推远。表面上的逻辑是，患者
资料包含临床语义，因此对齐后的音频表示应该更适合疾病预测。

问题在于 metadata 混合了三类完全不同的信息：

1. 真正可能从声音里听到的声学证据；
2. 患者背景和症状；
3. 数据来自哪个招募渠道、国家、平台或设备的 cohort 信息。

对比损失只知道“这一对属于同一个人”，不知道其中哪部分是疾病证据。只要年龄、症状或 cohort
能够帮助识别正确配对，loss 就会奖励模型学习它们。

## 2. Audit 先问了三个不同的问题

为了不再用一个下游 AUROC 混在一起回答所有问题，我们把 evaluation 拆成三层：

```text
模型有没有学会正确配对？       -> profile retrieval
它学到的是疾病还是患者／cohort？ -> age / sex / symptom / source probes
这些信息能不能跨人群预测疾病？   -> covariate-balanced matched evaluation
```

训练时只改变 audio--metadata pairing：

- `Correct`：音频与同一患者的 metadata 配对；
- `Within-label`：在相同疾病标签内部打乱患者，保留疾病／人群统计，破坏个体对应；
- `Global`：全局打乱，连标签共现也破坏。

如果 `Correct > Within > Global` 只出现在 retrieval，就说明模型学习了患者对应；只有这种优势
同时转化为 matched disease prediction，才有证据说 alignment 学到了可迁移疾病信息。

## 3. Audit 得到了什么

UKCOVID、CODA TB 和 Cambridge 的结果呈现出同一个核心形状：

- Correct pairing 的 profile retrieval 通常高于 Within-label，说明 alignment 真实发生了；
- Correct 表示更容易解码性别、年龄、症状或患者 profile；
- 但在协变量平衡的疾病评估中，Correct 相对 Within 的收益没有形成稳定、跨 backbone 的正向证据。

因此最准确的结论不是“对比学习没学到东西”，而是：

> **它学到的主要是 correspondence，但 correspondence 没有稳定变成 transfer。**

这也解释了为什么只报训练 loss、源域 AUROC 或 retrieval 会过度乐观：这些指标都可能在疾病
泛化没有改善时变好。

## 4. Repair v1：事后删除捷径为什么没有解决问题

第一种修复思路是在已经训练好的 Correct 表示中，找出能预测性别、年龄、来源和录音伪影的
线性方向，再把这些方向擦除，同时保留 raw audio 分支。

这个方法能降低部分录音属性的可解码性，但人口学和 cohort 信息并不是存放在几个独立方向里，
而是冗余地分布在表示中。Repair v1 没有稳定降低关键患者信息，也没有改善 matched disease
transfer。

这说明问题不只是 readout 看错了一两个方向；如果训练目标本身奖励患者／cohort 对应，事后线性
擦除很难把它变成疾病不变量。

## 5. Repair v2：从训练目标本身改变正负样本

第二种思路不再事后擦除，而是直接改变 disease contrast：

```text
positive：相同疾病、不同 cohort/environment
negative：不同疾病、相同 cohort，并且患者协变量尽量匹配
```

同时保留 raw audio：下游使用 `Raw + Repaired representation`，避免新目标把原本有用的声学信息
一起投影掉。

这个设计想迫使模型回答：“当 cohort 改变时，什么疾病信息仍然相同？”并阻止它仅靠年龄、
性别、症状或来源区分正负样本。

但是这种 loss 有一个经常被忽略的前提：训练数据里必须真的同时存在这两种 pair。如果某个环境
几乎只有阳性，另一个环境几乎只有阴性，就无法从同一份数据构造公平的 matched negative。

## 6. 为什么先跑数据门，而不是直接训练

我们在任何 Repair v2 embedding、prediction 或 matched-target score 产生之前，冻结了统一的
pair-support gate。每个训练 anchor 必须同时拥有：

- 至少两个同病、跨环境 positive；
- 至少一个异病、同环境、协变量可比的 negative。

总体及两个疾病标签各自都必须达到80% joint coverage。这个门槛不是自然定律，而是事前选定的
代表性要求：如果只能在少数特殊患者上定义 loss，就不能声称修复了整个训练分布。

## 7. 三个数据集的数据门结果

| 数据集 | cross-domain positive OK | strict negative OK | joint eligible | 决策 |
|---|---:|---:|---:|---|
| UKCOVID | 36.06% | 27.81% | **0.246%** | NO_GO |
| CODA TB | 100.00% | 67.50% | **67.50%** | NO_GO |
| Cambridge | 100.00% | 45.37% | **45.37%** | NO_GO |

三个数据集失败的方式不同：

- **UKCOVID**：招募来源与 COVID 标签近乎确定绑定。Test-and-Trace 没有阴性，REACT 几乎全是
  阴性；跨来源同病 positive 与同来源异病 negative 基本落在不同人群。
- **CODA TB**：七个国家都有 TB+/−，跨国家同病 positive 充足；但同性别、年龄接近、临床
  字段相近的同国家异病 negative 只覆盖67.5%，TB− 更只有60.5%。
- **Cambridge**：跨平台同病 positive 同样充足，但严格 matched negative 只覆盖45.4%；WEB
  source train 还有0个阴性、10个阳性的结构性空单元。

共同瓶颈不是找不到同病跨域样本，而是：

> **现有观察数据没有足够的“同一环境里，除了疾病外其余条件可比”的反事实对照。**

## 8. 这不是 Repair v2 的模型失败

Repair v2 从未进入训练，因此不能说方法有效或无效。数据门回答的是更早的问题：当前数据能否
公平地监督这个方法。答案是不能。

如果现在把80%改成60%、取消同环境限制，或者只训练 eligible minority，技术上都能跑出模型；
但科学问题已经变了。得到的结果可能再次利用 cohort shortcut，而且属于看到数据门结果后的
规则调整，不能作为原预注册假设的验证。

## 9. ICASSP 论文应该讲哪条故事

当前论文的主线不应改成“我们提出了一个成功的修复模型”，而应保持为：

> Clinical audio--metadata alignment has an evaluation ambiguity: successful participant
> correspondence does not imply transferable disease evidence. Pairing controls, nuisance
> probes and covariate-balanced evaluation are jointly required to reveal the gap.

中文就是：

> **临床音频与 metadata 对齐存在评价歧义：模型能够成功匹配患者，却不一定获得可迁移疾病
> 证据。只有把配对控制、信息探针和协变量平衡评估放在一起，才能识别这种差距。**

Repair v1 和 Repair v2 数据门适合放在 discussion／limitations：它们说明这个 failure mode 不是
靠简单线性擦除就能修复，而严格的 disease-invariant objective 又需要训练数据具有更好的共同
支持。这使 audit 更有操作价值，但不应把 mitigation 包装成正文正结果。

## 10. 真正的下一阶段

单个数据集内部继续修改 Repair v2 已经没有意义。若要做方法论文，应该重新建立一个具有完整
`environment × disease` 支持的训练集合，例如联合多个 COVID／呼吸疾病 cohort：

```text
UKCOVID + Cambridge + 其他临床 cohort
                  |
                  v
统一疾病定义、metadata schema 与音频预处理
                  |
                  v
跨数据集同病 positive + 数据集内 matched negative
                  |
                  v
新的、完全未用于设计的外部 cohort 测试
```

这已经不是当前论文里的“小修复”，而是新的 **multi-environment clinical contrastive learning**
项目。当前最合理的动作是先完成 audit paper；新的方法在新的数据支持和新 holdout 下独立冻结、
独立验证。

## 最后的故事线

```text
metadata alignment 看起来合理
        ↓
对比损失无法区分疾病证据与患者／cohort 信息
        ↓
Correct / Within / Global 证明模型确实学到患者 correspondence
        ↓
matched evaluation 发现 correspondence 没有稳定变成 disease transfer
        ↓
线性擦除 Repair v1 无法删除冗余 nuisance 信息
        ↓
Repair v2 试图从正负样本设计上学习 disease invariance
        ↓
三数据集 model-blind gate 发现单数据集缺少严格 matched counterfactual pairs
        ↓
结论：先正确审计 alignment；真正修复需要多环境数据，而不只是换一个 loss
```
