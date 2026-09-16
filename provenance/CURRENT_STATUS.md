# Current status at handoff

Date: 2026-09-11 (Asia/Shanghai)

## Achieved

- Repaired-F Train7000/Validation1000 data contract restored and verified for all 8000 samples.
- G0 FULL B8 and B16 completed for seeds 123/456/789 through epoch30.
- Validation checkpoint screen completed at epochs 15, 20, 23, 25, 26, 27, 28, 29, 30.
- Selected paper candidate: B8, epoch27; all three seeds retained.
- B8/seed789/e27 matched evaluation completed for FULL, NO_MESSAGE, NO_LAPLACIAN and OLD_ONLY.
- Validation metric records include per-device differential RMSE, nadir, maximum absolute RoCoF, terminal differential error, device-pair RMSE/MAE/sup error and spread errors across four time windows.
- Test/OOD/WECC reads during this selection and ablation work: zero.

## Training inventory notes

- FULL_B8: seeds123/456/789 complete30.
- FULL_B16: seeds123/456/789 complete30.
- NO_MESSAGE_B8: seeds123/456/789 complete30.
- NO_MESSAGE_B16: seeds123/456/789 complete30.
- NO_LAPLACIAN_B8: seed789 complete30; seeds123/456 absent.
- NO_LAPLACIAN_B16: seed123 complete30; seed456 stopped at13; seed789 absent.
- OLD_ONLY_B8: seeds123/789 complete30; seed456 absent.
- OLD_ONLY_B16: seeds123/456/789 complete30.

## Frozen decision

B8/e27 is the common Validation checkpoint. Do not change it based on Test. Seed789 is a representative seed with a balanced profile, while seed456 is stronger on some HIGH 2–10 s medians. Formal claims must use the three-seed result.

## Evidence classification

- Batch/epoch selection: complete for FULL Validation.
- F-message ablation: three-seed B8 evidence exists and is mixed by system/window.
- Laplacian ablation: one-seed B8 evidence; cross-seed confirmation missing.
- Spatial-loss ablation: seeds123/789 B8 exist, but current completed matched report covers seed789; seed456 training is missing.
- Test evidence for this selected model: not yet frozen in this handoff.
- Paper-ready universal superiority claim: not supported.
- Paper-ready conditional mechanism claim: supported on Validation, subject to cross-seed completion and final held-out confirmation.
