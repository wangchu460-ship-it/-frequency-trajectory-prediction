# G0 Validation checkpoint decision

## Decision

- Selected configuration: **FULL, batch 8, epoch 27**.
- Seeds: 123, 456, 789, evaluated with equal seed weight.
- Data: frozen Validation1000 only; D3 excluded from the primary summaries.
- Test/OOD/WECC reads: 0.
- Screened epochs: 15, 20, 23, 25, 26, 27, 28, 30. This was a staged local screen, not an exhaustive per-epoch search.

Epoch 27 is the best common checkpoint for the paper's scientific target: accurate device-relative trajectories, especially devices with large truth-defined spatial deviations. It is not the minimum for every metric.

## Core evidence (equal-seed mean of within-seed medians, Hz)

| Metric | B8 e26 | B8 e27 | B8 e28 | B16 e27 |
|---|---:|---:|---:|---:|
| IEEE39 HIGH differential RMSE, 2–10 s | 0.007524 | **0.007230** | 0.008860 | 0.009581 |
| NPCC140 HIGH differential RMSE, 2–10 s | **0.016014** | 0.016648 | 0.023876 | 0.024654 |
| NPCC140 HIGH differential RMSE, 0–2 s | 0.033966 | **0.031971** | 0.033097 | 0.031379 |
| IEEE39 all-pair RMSE, full horizon | 0.012586 | **0.012315** | 0.012769 | approximately 0.014 at its nearby checkpoints |
| NPCC140 all-pair RMSE, full horizon | 0.004832 | **0.004584** | 0.005425 | 0.005409 |

At epoch 27, B8 is about 24.5% lower than B16 for IEEE39 HIGH 2–10 s, 32.5% lower for NPCC140 HIGH 2–10 s, and 15.2% lower for NPCC140 full-horizon all-pair RMSE.

## Cross-seed stability

- IEEE39 HIGH 2–10 s: all three B8 seeds select epoch 27 among screened checkpoints.
- NPCC140 HIGH 2–10 s: seed-specific minima are 26, 27, 26. Epoch 27 is close to the B8 mean minimum at 26 (0.016648 versus 0.016014; +4.0%).
- The joint evidence therefore supports a single common epoch 27 rather than seed-specific checkpoint selection.

## Guardrails and limits

- IEEE39 HIGH 0–2 s continues improving through epoch 30; therefore epoch 27 does not maximize every early extreme endpoint.
- NPCC140 full-horizon HIGH differential RMSE also continues improving to epoch 30.
- BELOW_HIGH 0–2 s is generally best earlier (around epoch 15 for B8). Epoch 27 trades some low-deviation accuracy for materially better high-deviation and pair-relative fidelity. It should not be claimed to improve every device.
- `BELOW_HIGH` here is the complement of the frozen Train per-system Q90 HIGH definition. The deleted historical LOW-label artifact was not reconstructed or silently substituted.
- Epoch selection must remain frozen before Test inference.

## Formal interpretation

Use **B8/e27** for the principal G0 checkpoint and matched comparisons. Retain e30 only as a training endpoint/sensitivity result, not as the selected paper checkpoint.
