# UKCOVID 主实验：HeAR 第三骨干复核协议

冻结日期：2026-09-03  
状态：**在生成任何 UKCOVID HeAR embedding、alignment 或下游分数之前冻结**

## 地位与问题

UKCOVID 的 AST-6L 与 OPERA-CT 结果已经完成并读取。因此 HeAR 不是新的 confirmatory 实验，
而是专门面向健康声学预训练模型的 **post-hoc third-backbone robustness analysis**。

它只复核原问题：把同一 participant 的咳嗽表示对齐到临床 metadata，是否比同标签错配更能
建立 participant correspondence；这种差异是否能迁移到已经冻结的 covariate-balanced matched
与 matched-long 人群。

## 完全复用的部分

- `results/ukcovid_audio_cohort.csv` 的 72,458 人及原 train/validation/Standard/matched/
  matched-long 定义；不重切数据、不重新 matching；
- `results/metadata_texts.csv` 与 hash 已冻结的 Phi-2 text cache；
- Correct / Within-label / Global 三臂、seeds 0–4、500 epochs、batch 64、单向 InfoNCE、
  temperature 0.07、Adam lr 1e-3；
- 同 seed 三臂共享初始化、epoch 顺序和 dropout 流；只使用 epoch-500；
- source-validation 选择 readout 与校准，matched 数据不参与任何选择；
- 原疾病指标、profile retrieval、信息通道 probes、paired bootstrap 与解释纪律。

HeAR 的 512 维输入直接接到 `512 -> 1024 -> 2560` projector；除第一层输入宽度外，projector
结构与 AST/OPERA 完全相同。不得补零成 768 维，也不得为 HeAR 单独调参。

## 唯一变化：HeAR 音频表示

- 官方 `google/hear` SavedModel，revision
  `9b2eb2853c426676255cc6ac5804b7f1fe8e563f`，encoder 全冻结；
- 16 kHz mono float waveform，不做逐文件峰值归一化；
- 短音频右补零至 2 秒/32,000 samples；
- 长音频使用 `ceil(duration/2s)` 个均匀起点窗口，首窗从 0 开始、末窗贴住末尾，完整覆盖；
- 不使用 cough detector、能量筛选、标签或 metadata 选择片段；
- 一个 recording 的窗口等权平均；UKCOVID 每位 participant 已冻结为一条 cough recording；
- 输出必须为 participant-level `72,458 x 512`、有限、非零且顺序与 cohort 逐位一致。

长音频的 full-coverage mean pooling 是本项目的冻结适配，不是 HeAR 官方默认。它与 CODA HeAR
扩展使用完全相同的 waveform/window 定义。

## 执行与审计顺序

1. 验证既有 AST/OPERA 正式 manifest 和结果存在；
2. HeAR 纯窗口逻辑 self-test；
3. 分层 100 人 preflight，重复提取逐位一致；
4. 六个互斥 participant shards 并行提取，合并时核对 cohort/model/spec hash 和完整顺序；
5. train-only rehearsal；
6. Correct/Within/Global × 5 seeds × 500 epochs；
7. 复用既有 disease/probe evaluator；
8. 复用 profile retrieval 与 information-channel audit；
9. 只提交 aggregate JSON/Markdown，不提交 participant-level 产物。

TensorFlow HeAR 提取使用独立环境；PyTorch alignment/评测沿用原环境。UKCOVID 音频只在个人
服务器本地推理，不调用任何云端 API。

## 解释

- 与 AST/OPERA 同方向：只能写第三骨干事后稳健性方向一致；
- 不同方向：写 backbone-dependent；
- 不允许用 HeAR 结果替换、重定义或平均掉原来的主结论；
- matched 集已在先前研究中读取，所有 HeAR matched 数字均为探索性；新的确认仍需要未触碰
  外部数据。

