# FINAL_FROZEN_HELDOUT_EVALUATION_V2

All 13 frozen e<=60 models evaluated on Validation1000, Test1000 and all WECC1600. Test D3 retained. WECC usable1402 and risk198 reported separately. No WECC HIGH threshold.

FROZEN_CHECKPOINTS.json contains exact weights, hashes and Validation selection. A6 e56 and E3 e58 remain training-boundary unresolved. V1 held-out data were previously inspected; this freeze precedes V2 inference, not all historical held-out inspection.

ABSOLUTE_METRICS.csv and PAIRED_VS_G0.csv use one common metric definition and scene-to-group aggregation, separated by source system. Confidence intervals are nominal endpoint-wise 10000 group bootstraps, conditional on seed789. They do not measure training-seed variability or establish equivalence. No single-RMSE ranking.

Each model/split contains predictions.npz (prediction, target, mask, family, time, device rows) and SCENES.json for subsequent figures without inference. top-k uses centered RMS within each window with fixed k=1,3,5, and is omitted when fewer than k valid devices exist. Constant-vector Spearman values are undefined and paired coverage is explicitly reported.
