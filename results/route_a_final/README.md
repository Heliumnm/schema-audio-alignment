# Route A final auditable results

Date synchronized: 2026-08-23

Source workspace: `/mnt/hd/data_heliu/audio_provenance`

Frozen code lineage: `50ecdd9` (design), `fa0eb16` (implementation), `f17d37e`
(ranking/probability numerical separation), `a4d0a0a` (stable MLP scoring).

## What is stored here

This directory contains the compact JSON outputs needed to audit every number used in the
final ICASSP Route-A narrative. Files were copied byte-for-byte from the server and verified
against the server-side SHA-256 digests.

| directory | experiment | primary file(s) |
|---|---|---|
| `profile_retrieval/` | unique-profile correspondence manipulation check | AST and OPERA metrics |
| `information_channels/` | recording/patient/context/cohort/disease probe taxonomy | AST and OPERA metrics |
| `fusion_followup/` | metadata, raw audio, aligned audio, and raw-preserving fusion | AST and OPERA results |
| `probability_transport/` | prior correction, fixed shrink curve, target calibration transport | audio-only and fusion results |
| `mlp_readout/` | frozen nonlinear readout robustness | AST/OPERA results and score configs |
| `synthetic/` | preregistered controlled mechanism sweep and Phi-2 sensitivity | grid and Phi-2 summaries |

Large participant-level predictions, fitted MLP archives, embeddings, and checkpoints remain
on the server. They are intentionally not committed because they are large and may contain
participant identifiers. The compact JSONs preserve observed effects, confidence intervals,
per-seed effects, configuration hashes, prediction hashes, and gate verdicts.

## Frozen headline verdicts

- Retrieval: `correct > within-label > global` for both AST-6L and OPERA-CT; macro-profile
  `correct-within` MRR is +0.003175 [0.001605, 0.004917] and +0.003229
  [0.001545, 0.005000], respectively.
- Information channels: matched sex `correct-within` ΔAUROC is +0.1912 for AST and +0.1125
  for OPERA; COVID is +0.0062 and -0.0020, with both intervals crossing zero.
- Disease transfer: no stable cross-backbone, cross-readout matched disease-ranking gain;
  correct alignment does not outperform frozen raw audio.
- Probability transport: participant-disjoint target recalibration reduces audio-only
  `correct-within` Δ(-NLL) to approximately zero for both backbones, without creating an
  AUROC gain.
- Fixed MLP: does not rescue robust transfer; AST has a small fusion ranking effect that
  OPERA does not reproduce, while both have adverse source-calibrated NLL.
- Synthetic: correspondence gate passes, but high-confounding, dose-trend, and rho-zero
  boundary gates fail. `main_text_eligible=false`.

## MLP scoring technical correction

The original scorer used `object_array.tobytes()` for participant-string hashing, which
serializes process-local object pointers. Training and predictions were unaffected, and
every fit archive contained the correct participant sequence. Commit `a4d0a0a` added a
separate scorer that verifies the trainer hash, checks every participant ID value-for-value,
and records a stable length-prefixed UTF-8 cohort hash. No model was retrained and no metric
definition changed.

## SHA-256

```text
660d4b3e76bacd558d53ad2b82381a2ad07c3f80c529250ebe96aeab426922f8  fusion_followup/ast_results.json
3938b603ec5c9e93d4142375ea0f064eac32c67e4c1ace8bd1af3009d335a75a  fusion_followup/opera_ct_results.json
cb8857a0e40f5b92ffd87d57dc8fb77f4309b3a8d7d32ea3db7787ee202ca773  information_channels/ast_metrics.json
f52ec2140c2d0623912f37c2f3a7fb850d848a495d61329c1cf8741b64f8f905  information_channels/opera_ct_metrics.json
789aab3741d50c8557dc9dcba1a11573f94ce0c03383de79f126e341dabe9bae  mlp_readout/ast_results.json
69cae501d8c294d2131095d51f9b0e2aba51a8f7ce3795ee445d30744e66b896  mlp_readout/ast_score_config.json
5cf4279c54928f7cce7e6b73402a54dc9a85b59aa7f4e3c9041d5035bc8504d7  mlp_readout/opera_ct_results.json
cb581d85a9255e4c80752004fba63c6c037f64ce2e2424d6aa81533eb8708285  mlp_readout/opera_ct_score_config.json
abfef18b15fe9fa322687185429970ed7d857c2c705cca5c666a9b8c96de3897  probability_transport/audio_only_results.json
b8919abb2bec74eacd42b67711c578739e73806102124f0042392ffc37114455  probability_transport/fusion_results.json
e706e207e524c283cde8d536ad55f06ff27f1906455b8a606c74126f628e3207  profile_retrieval/ast_metrics.json
d44422ff8bef3997f7d7f73f4a8a5179718de4d76b006cb5e977f34451e959b0  profile_retrieval/opera_ct_metrics.json
026c4ee776c33990dc98d8b9c6a3d3dad6af14e675d872812d961e0192feab44  synthetic/grid_summary.json
1d3f483164ebcfca26bc8773277309ce1554cc012cb783cacc378e99d2047bcf  synthetic/phi2_summary.json
```

The scientific interpretation and claim boundaries are in
`docs/ICASSP_EXECUTION_STATUS_ZH.md`.
