# UKCOVID matched direct-fusion 结果

状态：**已完成；探索性 attribution control，不是外部确认。**  
冻结协议：`UKCOVID_MATCHED_DIRECT_FUSION_PREREG.md`。

## 这一步回答什么

前一阶段看到 metadata alignment 在 matched 人群中没有产生稳定的疾病增益，且
source-domain 校准迁移变差。但这不够区分两个解释：

1. alignment 特有地破坏了音频证据；
2. 原始音频在 metadata 之上本来就没有可检出的增量，alignment 只是没有救回来。

本实验用相同 schema metadata、相同线性头、相同 Standard train/validation 选择与
校准，在同一批 matched 参与者上比较 metadata only、raw AST direct fusion，以及
correct/within-label/global 三种 alignment fusion。

## 主 matched 结果（n = 1,814）

| arm | AUROC | NLL（越小越好） |
|---|---:|---:|
| metadata only | 0.5985 | 1.4341 |
| metadata + raw AST | 0.6013 | 1.4374 |
| metadata + correct alignment | 0.5973 | 1.4452 |
| metadata + within-label alignment | 0.5817 | 1.1753 |
| metadata + global-shuffled alignment | 0.5976 | 1.4207 |

配对比较：

- **raw AST direct fusion − metadata only**：ΔAUROC `+0.0028`
  `[-0.0019,+0.0071]`；Δ(−NLL) `−0.0034` `[-0.0194,+0.0126]`。两项均未排除 0。
- **correct − raw AST fusion**：ΔAUROC `−0.0040`
  `[-0.0093,+0.0015]`；Δ(−NLL) `−0.0077` `[-0.0301,+0.0157]`。两项均未排除 0。
- **correct − metadata only**：ΔAUROC `−0.0012`
  `[-0.0071,+0.0047]`；Δ(−NLL) `−0.0111` `[-0.0363,+0.0141]`。两项均未排除 0。
- **correct − within-label**：ΔAUROC `+0.0156`
  `[+0.0037,+0.0280]`，但 Δ(−NLL) `−0.2698`
  `[−0.3195,−0.2203]`，且五个 seed 全为负。

## 结论

主 matched 人群中，raw AST 在冻结的 schema metadata 之上没有可检出的稳定疾病增量；
因此此前的 OOD null 不能全部归因于 alignment。Correct metadata alignment 也没有优于
raw direct fusion，更没有优于 metadata only。

与此同时，correct 相对 within-label 呈现明确的 **ranking/calibration 分离**：AUROC 略高，
但 source-calibrated NLL 大幅更差。这与“正确逐人配对学到了源域对应关系，却把源域置信度
带入协变量平衡人群”一致，但不证明因果，也不能推广到非线性 readout 或其他数据集。

最准确的一句话是：

> 在 UKCOVID 的 matched 人群中，原始 AST 音频没有在 schema metadata 之上提供可检出的
> 配对疾病增量；正确 metadata alignment 未改变这一点，并相对 within-label 对照表现出
> 更差的 source-to-matched 概率迁移。

## 审计

- predictions SHA-256：`c4a5d52246c4b8818c6ac577c4932c85dbbdd433e996b69882c794777fb85d03`
- fitted heads SHA-256：`dfdeabb1f64f0b6be285eed2fb8a5bd6ec864ab0e017d8c42ca986af761c9d0b`
- frozen result SHA-256：`894b16b64cb6e7ccde3dbedb8442b85e6abc6f234e523be7179450bcca91bbfb`

限制：官方 UKCOVID test 已参与早期协议修正，因此本结果只能作为 discovery-cohort
attribution analysis；它不能替代外部患者级确认。
