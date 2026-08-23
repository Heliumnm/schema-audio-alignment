# Synthetic Correspondence--Transfer Stress Test（冻结设计稿）

日期：2026-08-20

冻结时状态：尚未实现、尚未运行。

> **执行结局（2026-08-23）：** 3×3 条件、每条件 10 seeds 的正式 sweep 已完成。
> correspondence gate 通过，但 high-confounding、dose-trend 与 rho-zero boundary gates
> 失败；Phi-2 sensitivity 也未取得正文资格。因此该实验不能作为一般机制证明。冻结设计原文
> 保留如下；结果见 `results/route_a_final/synthetic/`。

## 1. 定位

这个实验只回答：

> 当音频和 metadata 共享一个源域中与疾病相关、目标域中不再相关的 context signal 时，
> correct alignment 是否可能学到更强的逐样本对应，却没有获得可迁移的疾病收益？

它是 controlled mechanistic stress test，只证明“一种充分机制能够产生实证中观察到的
形状”。它不证明 UKCOVID 的真实因果机制，不是第二临床数据集，也不能证明实证结果
“不是 dataset artifact”。

## 2. 最小两因素生成模型

核心只保留两个因素：

- `D`：稳定 disease signal，只存在于音频，在 source 与 target 中都有效；
- `S`：音频与 metadata 共享的 shortcut/context，在 source 中与疾病相关，在 matched
  endpoint 中解除相关。

每个样本生成疾病 `y in {-1,+1}`。source prevalence 固定 `P(y=+1)=0.36`；matched
endpoint 固定 0.50。生成四维共享 context：

```text
s_j = rho*y + sqrt(1-rho^2)*eta_j,   eta_j ~ N(0,1), j=1..4
```

Metadata：

- 每个 `s_j` 用固定阈值 `{-0.43,+0.43}` 分成 low／mid／high；
- 得到 12 维 one-hot 并 L2 normalize；
- metadata 从不直接读取 `y`；
- 最多 `3^4=81` 个重复 profile，因此检索必须 profile-aware。

Audio `x in R^16`：

```text
x = [mu_d*y + eps_d,
     kappa*s_1 + eps_1, ..., kappa*s_4 + eps_4,
     eleven independent noise coordinates]
```

全部噪声为标准高斯。固定 `mu_d=0.37`，使第一维成为不进入 metadata 的稳定 disease
signal；其单独 oracle AUROC 约为 0.70。`kappa` 控制 shared shortcut 在音频中的可读强度。
所有域只使用 source-train 均值和标准差标准化。

这个最小模型故意不区分 patient shortcut 和 cohort shortcut。真实 UKCOVID 的 probe
taxonomy负责区分 sex／age／symptom／source／recording；synthetic 只验证“共享但不迁移的
context”这一充分机制。

## 3. 冻结参数网格

- source confounding：`rho_s in {0, 0.50, 0.95}`；
- shared shortcut audibility：`kappa in {0.5, 1.0, 2.0}`；
- 共 9 个训练 cell；
- 每个训练 cell 不重训地评估 `rho_t in {rho_s, rho_s/2, 0}`；
- 每 cell 10 个独立 simulation seeds；
- 每 seed：`n_train=4096`、`n_val=2048`、`n_source_test=4096`、每个 target endpoint
  `n_target=4096`；
- 结果后不得添加新的 rho、kappa、样本量、seed 或第三个 latent 因素。

## 4. Alignment arms 与训练

Arms：raw、correct、within-label、global。

- within-label 使用同疾病标签内的无 self-pair 固定 derangement；
- global 使用全体无 self-pair 固定 derangement；
- 每个 seed 的 mapping 在 500 epochs 中保持不变，与真实 UKCOVID 正式实验一致；
- 三臂共享初始化、audio batch order 和 dropout stream，只有 pairing 不同。

Projector：

```text
Linear(16,64) -> LayerNorm -> ReLU -> Dropout(0.1)
-> Linear(64,12) -> LayerNorm -> ReLU
```

训练固定为单向 audio→metadata InfoNCE、temperature 0.07、batch 64、`drop_last=False`、
Adam lr `1e-3` 默认参数、无 weight decay、无 scheduler、500 epochs，只使用最终 checkpoint。
正式前只允许 finite／non-collapse／手算 loss／pair hash smoke；smoke 失败即暂停，不因结果
修改架构。

## 5. 下游评测

冻结 representation 后，完全复用真实论文的线性 head：

- `C in {0.001,0.003,0.01,0.03,0.1,0.3,1,3}`；
- source validation one-SE 选择；
- source validation Platt；
- target 不选择、不校准；
- 额外固定 `[raw x, aligned z]` raw-preserving 线性头。

## 6. 指标

### 6.1 Disease transfer

- 主要：`rho_t=0` 的 paired `correct-within` Δ(−NLL)；
- 关键次要：paired ΔAUROC；
- 同时报告 source、`rho_t=rho_s/2` 与完整 shift curve；
- raw-preserving 判断稳定 D 是被 projector 压低，还是 raw audio 本来没有信息。

### 6.2 Correspondence gate

候选集合为所有出现的唯一 12 维 profile（最多 81 个），检索到相同 profile 即为正确。
主指标 MRR，R@1/R@10 辅助；报告 `CG=MRR(correct)-MRR(within)`。

### 6.3 Information probes 与真正的 use test

- 从 representation 解码 disease `y` 和四个 `s_j` 的三档；
- 报告每项 correct−within 与 within−global，不平均成总分；
- centered linear CKA 只作为 D/S 几何描述，不承担因果结论；
- 反事实 use test：固定 `y`、稳定 D 坐标与全部噪声，只重采样／翻转 S，测疾病 logit 的
  配对变化。生成变量已知，因此它比在 UKCOVID 上训练 adversarial eraser 更可解释；
- 不把 CKA、MRR、AUROC、NLL 混成单个 ATG；画 Correspondence--Transfer plane。

## 7. 理论操作预期

correct 的跨模态协方差可写成：

```text
Sigma_C = Cov(E[x|y], E[m|y]) + E[Cov(x,m|y)]
```

within-label 的两端在给定 y 后独立，因此：

```text
Sigma_W = Cov(E[x|y], E[m|y])
```

所以 `C-W` 隔离的是给定疾病标签后仍存在的逐样本 S correspondence。当 rho 与 kappa
增大时，correct 应提高 profile retrieval、S probe 与 source confidence；当 `rho_t -> 0` 时，
S 不再携带疾病排序，只剩未进入 metadata 的稳定 D 可以迁移。

## 8. 事前结果分支

- correct retrieval 不高于 within：alignment 没学会对应；机制实验不可解释，只能检查实现，
  不能调超参数救结果；
- 现象只在 `(rho_s=0.95,kappa=2)` 一个极端 cell 出现：只能称狭窄反例，不进入正文；
- correct 在 `rho_t=0` 稳定优于 within：failure-mode 假设被否定，必须如实报告；
- `rho_s=0` 仍明显恶化：更像 projector bottleneck／未共享 D 被对齐目标压低，不能归因
  confounding；
- raw-preserving 消除损失：结论改成 skip connection 可缓解，而非 alignment 必然有害；
- global 与 correct 相同：训练或 evaluator 无效，不能报告机制结论。

## 9. 正文准入门

只有同时满足以下条件，synthetic 才占 ICASSP 正文一张双面板图和约半页：

1. `kappa>=1` 时 CG 的 95% CI 全部高于 0；
2. 高 confounding 区域 source `C-W` ΔAUROC > 0，且 endpoint Δ(−NLL) < 0，CI 均排除 0；
3. regret 随 rho 或 kappa 呈预期剂量趋势，而非一个 cell；
4. `rho=0` 边界的整个 95% CI 落入预先 ROPE：`|ΔAUC|<0.005`、
   `|ΔNLL|<0.01`；
5. raw-preserving 给出与机制一致、可解释的边界。

任一关键门不通过，结果只进入 supplement 或完全不写；不得降低准入门。

## 10. Language-faithful sensitivity（仅一个 cell）

只在 `(rho_s=0.95,kappa=2)` 把最多 81 种 profile 序列化成与真实实验相同格式的 schema
字符串，用同一冻结 Phi-2 与 mask-aware mean cache 替换 12 维 one-hot metadata。其余训练、
arm、seed 和判据不变，不扩网格、不调参。

- 方向一致：说明 one-hot 机制在真实文本几何下仍可出现；
- 方向不一致：synthetic 只能称 objective-level sufficient mechanism，不能称
  audio-language 机制复现。

正文仍只展示核心 one-hot 网格；Phi-2 敏感性最多一句话或放 supplement。
