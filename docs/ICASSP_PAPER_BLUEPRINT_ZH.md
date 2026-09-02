# ICASSP 2027 最终论文蓝图：Pairing-Controlled Audit

日期：2026-08-23

状态：Route-A 实验已完成；本文件规定最终写作顺序与 claim 边界。

## 暂定题目

**Pairing-Controlled Auditing of Clinical Audio--Metadata Alignment: Separating
Participant Correspondence from Transferable Disease Evidence**

## 一句话故事

Clinical audio--metadata alignment 的评价经常把三种能力混在一起：学习逐患者对应、学习
标签／人群共现，以及学习可跨人群迁移的疾病证据。我们提出 correct／within-label／global
三配对审计并结合 matched evaluation。两个音频 backbone 都确实学到了患者 profile
correspondence，尤其是性别信息；但这种 correspondence 没有形成稳定、跨 backbone、跨
readout 的 matched COVID 增益。额外 NLL 主要来自目标域不支持的置信度，而非已证实的排序损失。

## 三层贡献

1. **问题贡献：** 明确区分 correspondence success、label/population association 与
   transferable disease evidence；source-domain accuracy 不能替代这三者的分解。
2. **方法贡献：** Pairing-Controlled Transfer Audit：`C-W` 隔离逐患者配对，`W-G` 隔离
   标签／群体共现，retrieval 验证 manipulation，taxonomy probes 解释信息通道，matched
   transfer 与 probability transport 检验泛化和置信度。
3. **科学发现：** AST-6L 与 OPERA-CT 均学到可测的 profile correspondence；correct pairing
   强烈保留 sex，部分保留 age/acquisition，但 COVID `C-W` 不稳定。线性／MLP、audio-only／
   fusion 均未得到 robust disease-transfer gain；correct 也不胜 raw audio。

第三点是 **UKCOVID discovery case study**，不是 metadata alignment 的普遍定理。

## Figure 1：Evaluation ambiguity 与三配对设计

上半部分画同一个 metadata positive pair 可能同时承载：

```text
audio <-> metadata
            |-- disease association
            |-- patient background
            |-- clinical context
            `-- cohort / protocol
```

下半部分画：

```text
correct       = label association + participant correspondence
within-label  = label association only
global        = neither

C-W -> participant correspondence
W-G -> label/population association
matched evaluation -> transfer
```

不要把 encoder architecture 画成 Figure 1 的视觉中心。

## Figure 2：Correspondence 与 transfer 分离

推荐双面板：

- Panel A：AST／OPERA macro-profile MRR，显示 `correct > within > global`；
- Panel B：matched `C-W` disease ΔAUROC 与 target-transport Δ(−NLL)，显示疾病收益不稳定、
  目标校准后 NLL 差约为 0。

不要把不同量纲合成 audit score。

## Table 1：模型真的学到了什么

正文只放 sex、age、recording、COVID 的紧凑 taxonomy：

| matched `C-W` ΔAUROC | AST-6L | OPERA-CT |
|---|---:|---:|
| sex | +0.1912 [0.1669, 0.2151] | +0.1125 [0.0943, 0.1311] |
| age 65+ | +0.0464 [0.0062, 0.0848] | −0.0004 [−0.0462, 0.0436] |
| loud recording | +0.0043 [−0.0187, 0.0271] | +0.0240 [0.0029, 0.0459] |
| COVID | +0.0062 [−0.0150, 0.0273] | −0.0020 [−0.0196, 0.0158] |

表下注明：raw audio 对这些 probe 的绝对可解码性通常更高；`C-W` 表示 alignment 相对控制臂
选择性保留什么，不表示比 raw 新增信息。

## Table 2：Transfer、readout 与 raw baseline

正文必须同时出现 raw audio、metadata、correct、within 和关键 fusion，防止 reviewer 认为
alignment 只是没和最强简单基线比较。

关键数字：

- raw audio 加到 metadata：matched ΔAUROC 为 AST +0.0028 [−0.0016, 0.0073]，OPERA
  +0.0041 [−0.0006, 0.0087]；
- 线性 fusion `M+C − M+W`：AST +0.0156 [0.0034, 0.0276]，OPERA +0.0021
  [−0.0068, 0.0107]；对应 NLL 两者均显著更差；
- `M+C − M+R` 的 AUROC 两个 backbone 都覆盖 0；correct 不胜 raw；
- 固定 MLP audio `C-W`：AST +0.0050 [−0.0154, 0.0270]，OPERA −0.0052
  [−0.0208, 0.0110]；
- 固定 MLP `C-R`：AST −0.0113 [−0.0291, 0.0068]，OPERA −0.0247
  [−0.0404, −0.0081]。

结论写成“not robust across backbone/readout and does not beat raw”，而不是“证明没有 disease
information”。

## Probability transport 的正文位置

只用一个紧凑段落封住 calibration 替代解释：

1. source-Platt matched `C-W` Δ(−NLL)：AST −0.0249，OPERA −0.0176；
2. prior-only correction 不消除负差；
3. 固定 shrink curve 显示越尖锐，AST 的 NLL regret 越大；
4. matched-long→matched target calibration 后：AST +0.00013
   [−0.00227, 0.00243]，OPERA −0.00020 [−0.00261, 0.00228]；
5. AUROC 仍分别为 +0.0062 和 −0.0020，区间覆盖 0。

因此准确句子是：

> The adverse source-calibrated NLL mainly reflected unsupported confidence under cohort
> shift; target recalibration removed the NLL gap but did not create a transferable ranking
> gain.

## Synthetic 的位置

Synthetic **不进入正文正向机制图**。冻结网格为 3×3 条件、每个条件 10 seeds，共 90 次模拟；
correspondence gate 通过，但 high-confounding、dose-trend、rho-zero gates 失败，Phi-2
sensitivity 也不具正文资格。

处理方式二选一：

- 最稳妥：正文限制中一句，详细失败门放 supplement；
- 如果版面允许：补充材料给完整 gate table，强调它否定了简单的一般机制解释。

不得说 synthetic 证明了 observed failure mode。

## 四页写作顺序

### 第 1 页：问题与 audit protocol

- 领域假设：alignment success 被默认等于 clinical representation；
- 漏洞：metadata 混合 disease／patient／context／cohort；
- Figure 1：三配对分解；
- 三项贡献。

### 第 2 页：冻结实现与 correspondence manipulation check

- UKCOVID patient-level cohort；
- AST-6L／OPERA-CT，Phi-2，audio-side projector；
- 五 seed、500 epochs、同初始化和 batch order；
- macro-profile retrieval；
- taxonomy probes、matched evaluation、paired bootstrap。

### 第 3 页：学到了什么，是否 transfer

- 先报 retrieval `correct > within > global`，证明训练有效；
- 再报 probe taxonomy；
- 最后放 matched transfer、raw baseline 和 fusion，这是 punchline。

### 第 4 页：替代解释与限制

- probability transport：NLL 是 confidence mismatch；
- MLP：不是简单线性 readout 限制；
- OPERA：不是 AST 单一 backbone；
- Coswara 与 CODA TB v1 数据门 NO-GO、CODA match-first v2 数据门 GO，以及 synthetic gate
  failure；必须明确数据门 GO 不是模型确认；
- 单个已完成 disease-transfer 数据集、test 参与过前期设计、尚无独立外部模型确认；
- 结论与最低审计建议。

## Abstract 必须包含的五件事

1. metadata alignment 的 evaluation ambiguity；
2. correct／within-label／global；
3. retrieval 证明两个 backbone 都学到了真实 correspondence；
4. sex correspondence 强、COVID transfer 不稳定、correct 不胜 raw；
5. source NLL 的主要问题是 unsupported confidence，并明确单数据集 discovery 限制。

## 最终 claim ladder

### 强证据支持

- correct pairing 学到了 participant/profile correspondence；
- sex correspondence 在两个 backbone 上稳定；
- matched COVID `C-W` 没有稳定跨 backbone/readout 增量；
- correct 没有超过 raw audio；
- source-calibrated NLL gap 可被 target calibration 基本消除。

### 只支持为边界条件

- age/acquisition 增量依赖 backbone；
- AST fusion 有小 ranking signal，但 OPERA 不复现；
- raw audio 在 matched-long 可能有很小增量，不能概括成完全没有音频信号。

### 不支持

- metadata alignment 普遍失败／普遍有害；
- 一般性的 confounding causal mechanism；
- 外部跨数据集确认；
- reasoning captions 或 audible-report alignment 有效；
- disease-invariant positive-pair 方法有效。

## 投稿前只做的事情

1. 从 `results/route_a_final/` 自动生成最终图表；
2. 将英文稿压到 ICASSP 四页并做数字一致性检查；
3. 补齐真实引用和 dataset/license 说明；
4. 做一次 claim audit，逐句检查是否超出上面的 ladder；
5. 不再增加 head、loss、backbone 或 post-hoc test 调参。

下一阶段 mitigation 方案见 `DISEASE_INVARIANT_ALIGNMENT_FUTURE_PLAN_ZH.md`，明确标记为未执行。
