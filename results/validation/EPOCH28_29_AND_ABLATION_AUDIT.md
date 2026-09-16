# Epoch 28/29 confirmation and matched ablation audit

## Frozen choice

The Validation-only checkpoint choice remains **FULL, batch 8, epoch 27**. Epochs 28 and 29 were evaluated for seeds 123, 456, and 789 with the same repaired-F data, initialization contract, masks, normalization, and metric implementation. D3 was excluded from primary summaries. Test/OOD/WECC reads were zero.

Equal-seed averages of within-seed medians:

| Metric (Hz) | e27 | e28 | e29 | Result |
|---|---:|---:|---:|---|
| IEEE39 HIGH differential RMSE, 2–10 s | **0.007230** | 0.008860 | 0.008367 | e27 best |
| NPCC140 HIGH differential RMSE, 2–10 s | **0.016648** | 0.023876 | 0.020220 | e27 best |
| NPCC140 HIGH differential RMSE, 0–2 s | **0.031971** | 0.033097 | 0.033648 | e27 best |
| IEEE39 all-pair RMSE, full horizon | **0.012315** | 0.012769 | 0.013411 | e27 best |
| NPCC140 all-pair RMSE, full horizon | **0.004584** | 0.005425 | 0.005764 | e27 best |
| IEEE39 BELOW_HIGH differential RMSE, 0–2 s | **0.011971** | 0.013897 | 0.014361 | e27 best |
| NPCC140 BELOW_HIGH differential RMSE, 0–2 s | **0.004880** | 0.005903 | 0.006055 | e27 best |

The late-neighbour check therefore confirms that epoch 27 is not an artefact of the coarse 25/30 screen. No further epoch inference is warranted.

## Matched structural ablation: NO_MESSAGE, B8/e27, three seeds

`NO_MESSAGE` removes the F-message contribution while retaining the Laplacian path. Positive delta means the ablation is worse than FULL.

| System / endpoint | FULL | NO_MESSAGE | Ablation delta | FULL paired win rate |
|---|---:|---:|---:|---:|
| IEEE39 HIGH, 2–10 s differential RMSE | 0.007230 | 0.009372 | **+30.1%** | 74.3% |
| IEEE39 full-horizon all-pair RMSE | 0.012315 | 0.013257 | **+7.5%** | 70.2% |
| NPCC140 full-horizon all-pair RMSE | 0.004584 | 0.005328 | **+16.2%** | 85.0% |
| NPCC140 HIGH, full differential RMSE | 0.035591 | 0.041566 | **+17.0%** | 72.5% |
| NPCC140 HIGH, 2–10 s differential RMSE | 0.016648 | 0.013981 | **−14.2%** | 38.0% |
| NPCC140 HIGH, 0–2 s differential RMSE | 0.031971 | 0.031593 | −1.1% | 55.4% |

Conclusion: removing F-message causes clear broad degradation in IEEE39 and in NPCC140 full-horizon pair/spatial fidelity. However, it improves the isolated NPCC140 HIGH 2–10 s median. The supported claim is therefore that F-message contributes to broad device-relative fidelity, **not** that it improves every system/window. The NPCC140 mid-window result is a real interaction/limitation that must be reported.

## Matched loss ablation: OLD_ONLY, B8/e27, seed 123 only

`OLD_ONLY` removes the added spatial auxiliary objective but leaves the architecture unchanged. Only seed 123 exists, so this is diagnostic rather than cross-seed evidence.

| System / endpoint | FULL | OLD_ONLY | Ablation delta |
|---|---:|---:|---:|
| IEEE39 HIGH, 2–10 s differential RMSE | 0.007279 | 0.007291 | +0.2% |
| IEEE39 HIGH, full differential RMSE | 0.027666 | 0.029025 | **+4.9%** |
| IEEE39 full-horizon all-pair RMSE | 0.012060 | 0.013035 | **+8.1%** |
| NPCC140 HIGH, 2–10 s differential RMSE | 0.015485 | 0.015874 | +2.5% |
| NPCC140 HIGH, full differential RMSE | 0.034402 | 0.029884 | **−13.1%** |
| NPCC140 full-horizon all-pair RMSE | 0.004972 | 0.004511 | **−9.3%** |

Conclusion: the spatial auxiliary objective helps the IEEE39 endpoints and slightly helps the NPCC140 HIGH 2–10 s endpoint, but OLD_ONLY is better on NPCC140 full-horizon spatial endpoints for seed 123. This ablation is mixed and single-seed; it does not support a universal loss-improvement claim.

## Evidence boundary

- The structural NO_MESSAGE result is a valid three-seed matched ablation, but its effect is system/window dependent.
- The OLD_ONLY result is one-seed evidence only.
- No B8 NO_LAPLACIAN checkpoints exist; no result was inferred from B16 or another seed.
- `BELOW_HIGH` is the complement of the frozen Train system-specific Q90 HIGH threshold, not the deleted historical LOW label.
