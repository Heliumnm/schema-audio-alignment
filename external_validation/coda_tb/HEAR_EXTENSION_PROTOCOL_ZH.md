# CODA TB：HeAR 第三骨干扩展协议

冻结日期：2026-09-03  
状态：**在生成任何 CODA HeAR embedding、projector、retrieval、probe 或 TB 分数之前冻结**

## 1. 科学地位

HeAR 是在 AST-6L 与 OPERA-CT 正式结果已经读取后新增的第三骨干，因此本实验只能作为
**post-hoc backbone robustness extension（事后骨干稳健性扩展）**，不能被写成预注册的独立外部
确认。它不改变 AST/OPERA 的既有结果、判据或解释分支，也不把三个 backbone 合并挑选最有利
结果。

本扩展只问：在同一冻结队列、同一文本、同一 Correct/Within-label/Global 配对和同一评测链下，
换成专门为健康声学预训练的 HeAR 后，是否仍观察到“participant correspondence 可以建立，但
matched disease transfer 未必随之改善”的分离。

## 2. 不变项

以下内容逐项复用 `PREREGISTRATION_FORMAL_MODELS_V1_ZH.md`，不得因 HeAR 结果修改：

- CODA match-first-v2 的 1,081 名 participant、9,749 条 solicited cough、split 和 100 个 matched
  pairs，以及三个受控输入 SHA-256；
- 16 字段 metadata schema、冻结 Phi-2 文本 cache；
- `correct`、`within_label`、`global` 三种 train-only 配对；
- seeds `0,1,2,3,4`，同 seed 三臂共享初始化、batch 顺序和 dropout 随机流；
- 单向 audio-to-text InfoNCE、temperature 0.07、batch 64、Adam lr 1e-3、500 epochs，只使用
  epoch-500；
- profile retrieval、source-only readout/校准、matched pair × seed bootstrap、probes、主比较、
  等价区间和解释分支；
- participant 是训练、评测和 bootstrap 单位，每位 participant 只贡献一个 audio representation。

Projector 仍为 `input_dim -> 1024 -> 2560`，两层之后均为 LayerNorm + ReLU，第一层之后 dropout
0.1。HeAR 的 `input_dim=512` 由冻结 embedding cache 的第二维断言得到；不得人为扩成 768 维。

## 3. HeAR 唯一变化

- 官方模型：`google/hear`，revision
  `9b2eb2853c426676255cc6ac5804b7f1fe8e563f`，本地 SavedModel；
- 推理固定使用独立 TensorFlow 2.18 环境，与 Google 当前 HeAR 支持仓库一致。TensorFlow 2.15
  虽能读取 SavedModel signature，但实际前向因 VHLO/StableHLO 版本不兼容而失败；本修正发生
  在任何 embedding 生成之前，不改变样本、窗口、聚合或评测协议；
- 该 TensorFlow wheel 编译要求 cuDNN 9.3；服务器系统 cuDNN 9.1 无法执行卷积，因此在同一
  隔离环境安装 `nvidia-cudnn-cu12==9.3.0.75`，运行脚本显式把这份动态库置于搜索路径首位。
  真实 2 秒零波形前向已验证输出为有限的 `(1, 512)`，此为纯运行时兼容修复；
- encoder 完全冻结，官方输出为每个 2 秒片段一个 512 维 embedding；
- 输入为 16 kHz mono float waveform，不做逐文件峰值归一化；
- 短于 2 秒：右侧补零至 32,000 samples；
- 长于 2 秒：使用 `ceil(duration/2s)` 个均匀起点窗口，首窗从 0 开始、末窗贴住音频末尾，完整
  覆盖首尾；
- 不使用 cough detector、能量阈值、content selection、标签或 metadata 选择窗口；
- 先等权平均同一 recording 的全部窗口，再等权平均同一 participant 的全部 recordings。

这与官方 HeAR 的两秒输入定义一致，但长音频的 deterministic full-coverage 聚合是本项目为公平
participant-level 比较预先固定的适配，不声称是 HeAR 官方默认聚合。

## 4. 技术门与顺序

1. 先验证 AST 与 OPERA 的正式 aggregate result 和训练审计均已存在，避免 HeAR 反向改变主实验；
2. 使用独立 TensorFlow 环境加载 HeAR；受控 CODA 音频只在本地服务器处理，不调用云 API；
3. 对冻结 preflight participant 重复提取，要求逐位一致、512 维、有限、非零；
4. 完整覆盖 1,081/1,081 participant；任何失败必须停止，不得删除 participant 或重新 matching；
5. 用原 PyTorch 环境做 seed-0、50-epoch train-only S1，只检查 loss 有限且下降、输出不坍缩；
6. 三臂 × 五 seeds × 500 epochs；
7. 技术审计必须核对完整 run set、512 输入维度、配对合法性，并证明所有 pairing hash 与既有
   AST/OPERA 对应 run 相同；
8. 最后运行既有 CODA evaluation，一次性产生 HeAR aggregate JSON。

TensorFlow 与 PyTorch 分环境运行：TensorFlow 只写 participant-level NumPy embedding cache，
PyTorch 只读取该 cache。任何原始音频、participant 表、embedding、prediction、checkpoint 或
pairing 均不得提交 GitHub。

## 5. 解释纪律

- HeAR 内部仍以 `Correct-Within` retrieval 判断 correspondence，以 matched-target 上配对
  `delta(-NLL)` 与 `delta AUROC` 判断 transfer；
- HeAR 若与 AST/OPERA 同方向，只能写“第三骨干事后稳健性分析方向一致”；
- HeAR 若不同，必须写 backbone-dependent，不把更有利的 backbone 提升为主结果；
- HeAR 在其原始论文中具有 COVID/TB 等下游结果，并不保证在本项目的冻结配对目标或 matched
  population 上得到正迁移；
- matched target 已在前两个 backbone 中读取过，因此 HeAR 的所有 matched 结果都是探索性，
  需要新的 untouched 数据才能确认。
