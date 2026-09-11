# Pairing-Controlled Audit 三项补充实验：冻结分析协议

日期：2026-09-11  
状态：**新增结果尚未产生；本文档在 E1--E3 执行前冻结。**

冻结基线 commit：`70e96794f8c0ea19b9bebdd94423c757a1f91199`  
UKCOVID cohort SHA-256：`974662e152d3b4b56765e3bbfa925a5a500f49c6d35d96410c1f8041b786a128`  
Metadata text manifest SHA-256：`f295553bd65165037fa90e75b681cb50e6c08fbef6ad06435b9139308ae7e585`

## 0. 定位与证据边界

这三项分析是看到 UKCOVID 主结果之后提出的定向诊断，不是独立确认实验。它们分别回答：

1. 现有 profile retrieval 是否主要依赖性别或疾病标签带来的粗粒度候选差异；
2. 原来的 `Correct - Within-label` 差异中，有多少在进一步保留记录性别后仍然存在；
3. matched disease 结果不明确，是不是主要受 source-trained readout 限制。

三项分析均不得：

- 改动已经冻结的 C/W/G projector；
- 用 matched 结果重新选择 backbone、projector、epoch 或 metadata schema；
- 把 probe 差异写成疾病分类器对该属性的因果依赖；
- 把 target-assisted readout 混入 source-only 主结果；
- 用新增分析把置信区间跨零改写成“等效”或“完全没有信息”。

建议执行顺序为 **E1 → E2 → E3**。E1 不需要训练；E2 训练新对照；E3 等 E2 完成后一次性覆盖
`raw / C / W / G / W_{y,s}`，避免重复运行 target-assisted readout。

---

## 1. E1：条件化 profile retrieval

### 1.1 问题

> Correct 的 profile-retrieval 优势，在排除容易利用的性别或标签相关候选差异后，是否仍然存在？

本实验只重算检索，不重新训练 projector。输入为 UKCOVID 两个主 backbone、三个 pairing arm、
五个 seed 的冻结 validation projection 和同一份冻结 Phi-2 profile bank。

### 1.2 四个候选库

对每个 validation query 分别构造：

1. `unrestricted`：完整唯一 profile bank；
2. `same_sex`：候选 profile 至少对应一名与 query 记录性别相同的 validation participant；
3. `same_label`：候选 profile 至少对应一名与 query 疾病标签相同的 participant；
4. `same_label_and_sex`：候选 profile 至少对应一名同时满足同标签、同性别的 participant。

候选资格只由冻结 validation manifest 决定，不依赖 arm、seed、相似度或检索结果。同一 query 在
C/W/G 中必须使用逐位相同的 candidate ID 列表。疾病标签仅用于限制评测候选，不进入文本编码器。

### 1.3 重复 profile 与 query 资格

- 相同 schema 仍折叠为一个语义候选，不因其对应多人而复制；
- 一个 profile 若在多个标签或性别组出现，可进入相应多个 query-conditioned bank，但在单次检索中
  仍只出现一次；
- query 的真实 profile 必须在候选库中，否则该 query 对所有 arm/seed 同时排除；
- 每个候选条件报告 query 数、唯一 profile 数、每个 query 的候选数分布以及覆盖率；
- 不允许只在某个 arm 排除 query。

### 1.4 指标与随机基线

主指标仍为 macro-profile MRR，micro MRR、R@1 和 R@10 为辅助。每个真实 profile 内先平均 query
reciprocal rank，再对 profile 等权平均。主要差值为：

\[
\Delta_{\mathrm{pair}}^{(c)}=
\operatorname{MRR}^{(c)}(C)-\operatorname{MRR}^{(c)}(W),
\]

其中 \(c\) 表示候选条件。另报 `C-G` 与 `W-G`，但不改变主比较。

候选库大小因 query 而异，所以每个条件必须报告自己的随机排序基线。对含 \(K_i\) 个候选的 query，
随机 MRR 期望为 \(H_{K_i}/K_i\)，R@k 期望为 \(\min(k,K_i)/K_i\)；macro 随机基线按相同的
profile 等权方式汇总。

置信区间沿用现有 profile × seed 配对 bootstrap，10,000 次，RNG `20260820`。C/W/G 的 query、
seed 和 profile 必须保持配对。

### 1.5 冻结解释

- 条件化后 \(\Delta_{\mathrm{pair}}>0\)：性别或标签的粗粒度候选过滤不足以解释全部 correspondence；
- 条件化后效应缩小或跨零：检索结果对候选构成敏感；这不能单独证明疾病分类器使用了性别；
- 不直接比较不同候选库的原始 MRR，也不把 MRR 的缩小比例解释为“性别贡献比例”；
- 若候选库过小或覆盖不足，只报告可行性，不补充放宽条件。

计划脚本：`src/eval_conditional_profile_retrieval.py`  
计划输出：`results/pairing_followup/e1_conditional_retrieval.json` 和逐 query rank 文件。

---

## 2. E2：标签与性别共同保持的 Within 对照

### 2.1 新训练臂

定义：

\[
a_i\leftrightarrow m_j,\qquad
y_i=y_j,\quad s_i=s_j,\quad i\neq j,
\]

记为 \(W_{y,s}\)。其中 \(s_i\) 是数据中记录的性别字段，包括显式 missing 类别。该臂在保留
疾病标签和记录性别一致性的前提下打乱 participant profile。

主要比较为：

\[
C-W_{y,s},\qquad W_{y,s}-W.
\]

前者检查更严格控制性别后，精确配对增量还剩多少；后者检查仅保留性别匹配关系相对普通
Within-label 增加了什么。两者均为受控训练对比，不是严格因果分解。

### 2.2 运行前支持度审计

冻结的 UKCOVID Standard train 共 20,714 人。按 `COVID label × recorded sex` 分层：

| COVID | sex | n |
|---:|---|---:|
| 0 | FEMALE | 7,378 |
| 0 | MALE | 5,866 |
| 1 | FEMALE | 4,690 |
| 1 | MALE | 2,775 |
| 1 | MISSING | 5 |

五个实际层均至少有两人，因此可以在原 20,714 人训练集合上构造无自配对的层内置换。E2 不删除
参与者，也不重训既有 C/W/G。正式新增训练量固定为：

\[
2\ \text{backbones}\times5\ \text{seeds}=10\ \text{projectors}.
\]

如果执行代码复核得出与上表不同的层计数，立即停止；不得临时跨性别配对、回退到自配对或只为
新臂删除样本。只有另行冻结共同支持集合并在其上重训 C/W/G/\(W_{y,s}\) 四臂，才允许改走
40-projector 方案。

### 2.3 配对构造与随机性

- 每个 `(label, sex)` 层先用独立 pairing RNG 打乱 participant indices，再循环平移一位；
- n=2 的层直接交换；所有层必须满足 `pair[i] != i`；
- 每个 seed 的 pairing 固定一次并保存 participant-to-participant mapping 与 SHA-256；
- pairing RNG 与模型初始化、minibatch 和 dropout RNG 分离，避免仅因生成配对而改变训练随机流；
- 同 seed 的 \(W_{y,s}\) 必须复用既有 C/W/G 的初始化、audio batch 顺序、dropout stream、
  optimizer、500 epochs 和 final-checkpoint 规则。

### 2.4 冻结 endpoints

按现有协议计算：

1. source-validation macro-profile MRR；
2. matched 的 sex、age、symptom、cohort/acquisition 和 disease probes；
3. matched disease AUROC、NLL、Brier 与 calibration；
4. frozen raw audio reference；
5. `C-W_{y,s}` 与 `W_{y,s}-W` 的 participant/profile × seed 配对区间。

最关键的表示诊断是 sex probe：

- 若 `W_{y,s}-W` 明确为正且 `C-W_{y,s}` 接近零，原 C-W sex 结果主要依赖记录性别匹配关系；
- 若 `C-W_{y,s}` 仍明确为正，记录性别不足以解释全部精确配对差异；
- 无论哪种形状，都不能据此声称 disease classifier 因果使用性别。

疾病结果仍需同时比较 `C-W_{y,s}` 和 `C-Raw`。仅超过 \(W_{y,s}\) 而不超过 raw audio，不能
写成 alignment 改善疾病表示。

计划脚本：`src/train_within_label_sex_control.py`  
计划输出：pairing manifests、10 个 final projector hashes、retrieval/probe/disease summary 与逐参与者预测。

---

## 3. E3：target-assisted diagnostic readout

### 3.1 问题

> 当前疾病结果不明确，是表示本身没有提供明显优势，还是 source-trained readout 没有适应目标分布？

固定所有音频表示，在 participant-disjoint matched-long 上训练和选择疾病 readout，然后原样应用到
matched。matched 不参与标准化、正则化、校准或任何模型选择。

该实验使用目标分布标签，因此必须标为 **target-assisted diagnostic**。matched-long 已用于此前的
校准诊断，matched 结果也已在主分析中读取；E3 是事后机制诊断，不是 untouched confirmation，不能
替代 source-only 主结果，也不能用于重新选择 projector。

### 3.2 冻结 readout 协议

沿用主分析的正则化 logistic readout family 和 C 网格：

1. 只在 matched-long 上拟合 feature mean/scale；
2. 在 matched-long 内做 participant-level、按 disease label 分层的五折交叉验证；
3. 仍用 one-standard-error rule 选择更强正则，不根据 matched 表现选 C；
4. 用 matched-long 的 out-of-fold logits 拟合一维 Platt calibrator；
5. 选定后在完整 matched-long 上重新拟合 scaler 与 logistic readout，再把 calibrator 原样应用到
   matched；
6. 所有表示共用相同 folds、C grid、选择规则和 bootstrap indices。

主诊断指标是 matched 上的 paired disease ΔAUROC；Δ(−NLL)、Brier 和 calibration slope 为辅助。
每个 backbone 分别报告，不跨 backbone pooling。

### 3.3 运行规模

在 E2 之前，两个 backbone 的 `Raw + C/W/G × 5 seeds` 共：

\[
2\times(1+3\times5)=32\ \text{个最终 readout}.
\]

E2 加入 \(W_{y,s}\) 后再增加 10 个，因此本轮建议一次性运行 42 个最终 readout。交叉验证中的临时
拟合不计入该数。Raw 为确定性表示，每个 backbone 只拟合一个最终模型；复制 seed 轴只用于配对
统计，不能冒充五次独立训练。

### 3.4 冻结比较与解释

依次报告：

1. `C-W` 和 `C-W_{y,s}`；
2. `W_{y,s}-W`；
3. `C-Raw`；
4. 与原 source-trained readout 的同表示、同 endpoint 差异。

- target-assisted 下 C-W 变为稳定正向：说明 disease 结论依赖 readout 的训练分布；仍需检查
  C-Raw，且不能把它写成 source-only 可部署收益；
- C-W 正而 C-Raw 不正：说明精确配对相对 shuffle 更易被目标 readout 利用，但没有证明 alignment
  优于原始音频；
- target-assisted 下仍无稳定优势：在这套固定线性诊断中没有找到隐藏收益；不能证明表示中完全没有
  disease information；
- 若两 backbone 方向不同：报告 backbone-dependent boundary，不挑选较好的一条作为结论。

计划脚本：`src/eval_target_assisted_readout.py`  
计划输出：`results/pairing_followup/e3_target_assisted_readout.json`、逐参与者逐 seed 预测、fold 和
模型选择记录。

---

## 4. 论文放置规则

- **E1** 是 retrieval 结果的必要敏感性分析，优先放 supplementary；正文最多一句总结；
- **E2** 是最重要的新训练控制。如果它实质改变 sex 或 disease 的解释，应进入正文主结果；
- **E3** 永远标为 target-assisted diagnostic，只能放 supplementary/Discussion，不能与 source-only
  主结果并列表述为同等级证据；
- 三项都保留完整负结果，不因某个候选条件、backbone 或 readout 更好而改主指标；
- 在 E1--E3 完成前，当前四页稿的已有结论保持不变，未运行计划不得写成结果。

## 5. 最终新增结果表模板

| Analysis | Backbone | Primary contrast | Estimate | 95% CI | Interpretation |
|---|---|---|---:|---:|---|
| E1 same-sex retrieval | AST-6L | C-W macro MRR | pending | pending | pending |
| E1 same-label+sex retrieval | OPERA-CT | C-W macro MRR | pending | pending | pending |
| E2 sex-controlled pairing | AST-6L | C-Wys sex ΔAUROC | pending | pending | pending |
| E2 disease transfer | OPERA-CT | C-Wys disease ΔAUROC | pending | pending | pending |
| E3 target-assisted | AST-6L | C-W disease ΔAUROC | pending | pending | diagnostic only |
| E3 target-assisted | OPERA-CT | C-Raw disease ΔAUROC | pending | pending | diagnostic only |
