# B8 seed789 matched FULL/ablation evaluation

All four runs have 30 model snapshots and COMPLETE metadata. The comparison uses the previously selected epoch27 for every arm. Newly uploaded NO_LAPLACIAN and OLD_ONLY passed strict checkpoint loading. Checkpoint hashes match the inference manifest. Configurations differ only in arm and, for OLD_ONLY, lambda_early (1.7879867681635668 to zero). Initial-state hash, batch, seed, learning rate, normalization, repaired-F contract and runtime match. All metric record keys align exactly.

Validation1000 was evaluated. Primary results exclude D3; HIGH follows the frozen Train system-specific centered-trajectory RMS threshold. BELOW_HIGH is its complement, not the previously deleted LOW stratum. No Test/OOD/WECC inference was performed.

## Spatial evidence

Values below are medians of per-device differential trajectory RMSE (Hz), not an error of a system-average trajectory.

| System / HIGH window | FULL | No F-message | No Laplacian | OLD_ONLY |
|---|---:|---:|---:|---:|
| IEEE39 0–2 s | 0.093551 | 0.093759 | 0.093633 | 0.093780 |
| IEEE39 2–10 s | 0.007388 | 0.007285 | 0.007609 | 0.008161 |
| IEEE39 10–30 s | 0.002837 | 0.005508 | 0.004570 | 0.003229 |
| IEEE39 full | 0.026849 | 0.026945 | 0.027658 | 0.028701 |
| NPCC140 0–2 s | 0.031440 | 0.031554 | 0.033725 | 0.033579 |
| NPCC140 2–10 s | 0.019745 | 0.013233 | 0.022947 | 0.024498 |
| NPCC140 10–30 s | 0.011351 | 0.010546 | 0.012365 | 0.011037 |
| NPCC140 full | 0.034644 | 0.041469 | 0.032668 | 0.034317 |

Compared with FULL, removing Laplacian worsens NPCC HIGH early/mid medians by 7.3%/16.2%; removing the auxiliary loss worsens these by 6.8%/24.1%. In the mid window FULL wins for 73.1% of paired HIGH devices against No Laplacian and 66.4% against OLD_ONLY. These support targeted spatial supervision and the Laplacian contribution for that response phase.

F-message has mixed effects: removal worsens NPCC HIGH full-horizon median by 19.7%, with FULL winning 75.1% of paired devices, but improves its 2–10 s median by 33.0%. In that mid window FULL wins only 31.8% of devices. This is a substantial counterexample to any universal message-benefit claim.

## Pair-relative and low-deviation tradeoffs

| Endpoint | FULL | No F-message | No Laplacian | OLD_ONLY |
|---|---:|---:|---:|---:|
| IEEE39 full pair RMSE, scenario median | 0.011480 | 0.011514 | 0.011935 | 0.011062 |
| NPCC140 full pair RMSE, scenario median | 0.004437 | 0.004601 | 0.004256 | 0.004242 |
| IEEE39 BELOW_HIGH 0–2 s differential RMSE | 0.011705 | 0.012045 | 0.011577 | 0.010979 |
| NPCC140 BELOW_HIGH 0–2 s differential RMSE | 0.004232 | 0.004733 | 0.004047 | 0.002502 |

The auxiliary loss shifts the accuracy tradeoff toward high-deviation early/mid responses; OLD_ONLY reduces NPCC BELOW_HIGH early median error by 40.9%. Therefore the loss cannot be described as improving all devices or preventing all false spatial variation.

Differences of medians and medians of paired differences are different statistics. The JSON includes both, plus paired win rates and P90. For example IEEE39 mid No F-message has a slightly lower marginal median while FULL improves 70.1% of paired devices. This is not a contradiction and neither statistic alone establishes universal improvement.

## Nadir and maximum absolute RoCoF

NPCC HIGH full-horizon nadir error increases 17.9% without F-message, 4.4% without Laplacian and 3.1% with OLD_ONLY. FULL wins 85.0%, 54.9% and 55.2% of paired devices, respectively. This reinforces the F-message contribution to nadir accuracy for this seed.

Maximum absolute RoCoF does not show universal structural gains: removing either physical operator slightly improves its median error in both systems. OLD_ONLY worsens NPCC RoCoF median by 3.4%, with FULL winning 94.8% of paired devices, but does not show the same advantage on IEEE39. Do not claim all-extreme-metric superiority.

## Paper judgment

PARTIAL_SUPPORT: sufficient for a transparent Validation ablation subsection demonstrating system- and time-dependent contributions; insufficient alone for a finalized universal superiority claim.

Supported narrative: the physical message/Laplacian model with targeted spatial supervision improves selected high-deviation device-relative response regimes. The two operators contribute differently across systems and time windows, and targeted supervision carries an accuracy tradeoff for lower-deviation devices.

Unsupported narrative: every ablation degrades all windows; both physical operators are always complementary; the auxiliary loss improves every device; maximum RoCoF is universally improved; seed789 is objectively the best seed.

Epoch27 and seed789 were selected with Validation evidence. Thus this comparison is a matched-budget descriptive ablation, not independent confirmatory evidence or an equal-search-budget best-checkpoint comparison. The earlier description of seed789 as uniquely best was too strong: it is a chosen representative with metric tradeoffs. Confirmatory paper results should retain all seeds and use the frozen protocol on held-out data. No confidence intervals or statistical-significance claims are made here. NO_LAPLACIAN currently has only one matched B8 seed; additional seeds would be necessary for its cross-seed claim.

## Outputs

B8_S789_MATCHED_AUDIT.json includes protocol differences and all available per-system, HIGH/BELOW_HIGH, SG/GFM/GFL, pair-family, window, nadir, RoCoF, terminal and spread summaries with P90 and paired differences. Underlying per-device/per-scenario records remain in each epoch027 metrics.jsonl.gz.
