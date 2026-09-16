# V2 common evaluation protocol

Checkpoint selection is frozen in `FROZEN_CHECKPOINTS.json` before V2 inference. The four historical HIGH group-macro endpoints and their reference values are unchanged. Candidate centers are e3–58 within the e1–60 horizon. Selection lexicographically minimizes five-epoch maximum, five-epoch mean, center score, then epoch. No held-out result enters selection. Earlier V1 held-out results have already been viewed; this chronology is disclosed rather than described as a globally untouched test set.

All models use the same frozen manifests and training normalization. A6 changes only the relation source to PRE-event. A5 disables the physical relation path; it is not a newly trained “device-only” architecture. E1/E4 use the historical frozen base graph belonging to each manifest record. Checkpoint loading is strict.

Test primary population is all 1,000 frozen records. EXPLICIT_D3, LABELED_NON_D3 and UNLABELED are sensitivity strata. The historical Validation selection excluded explicit D3 records; this exclusion is retained only when reproducing the selection endpoints. V2 descriptive Validation tables report all 1,000 records.

WECC inference loads all 1,600 records. FULL_TRAJECTORY_USABLE and MASKED_TRAJECTORY_USABLE jointly form the 1,402 primary cases. The 198 PHYSICAL_RISK_DIAGNOSTIC_ONLY cases are a separate diagnostic population. No target-derived or source-transplanted WECC HIGH threshold is used.

Time windows are [0,30], [0,2], [2,10] and [10,30] seconds, with inclusive endpoints and trapezoidal weights computed within each window. Existing frozen masks apply identically to all predictions. Centering subtracts the masked arithmetic device mean at each time, separately for prediction and truth. Device differential RMSE is the weighted RMS of the difference between these centered trajectories. Source HIGH labels use full-trajectory true centered RMS and frozen thresholds IEEE39=0.0184240117377 and NPCC140=0.012875054017 Hz.

Cross-family pair-relative RMSE evaluates prediction error in y_i-y_j, using the pairwise intersection of masks. Each scene's family-pair result pools its valid pair/time squared errors before taking the square root, matching the existing spatial_metrics implementation. It is not a difference between family-mean trajectories.

Device identification ranks the true and predicted within-window centered RMS. Report tie-aware Spearman correlation and top-k set recall for fixed k=1,3,5; deterministic device-index order breaks top-k ties. Omit a k when fewer than k devices have positive valid time weight. Constant vectors have undefined Spearman, with paired coverage explicitly reported. Spatial amplitude includes true and predicted masked RMS plus mean absolute per-device amplitude error.

Nadir diagnostic is absolute difference of minima on valid points within each window. RoCoF diagnostic is absolute difference of maximum absolute adjacent finite-difference derivatives, using only adjacent valid samples. This is a derivative diagnostic at the saved sampling resolution, not a continuous-time extremum guarantee.

Aggregate devices to scene means, then scenes to prospective/input-group means, then give groups equal weight. Report each source system separately. Bootstrap the paired group differences 10,000 times with seed 20260912. Positive error difference means the comparison model has higher error than G0; higher rank correlation/recall is favorable. Report paired group median difference and group win rate. Undefined rank correlations are accompanied by coverage counts; error endpoints require exactly matched record sets.

Intervals are nominal endpoint-wise 95% intervals without multiplicity correction. They quantify held-out group variation conditional on seed789, not training-seed uncertainty. A confidence interval spanning zero does not establish equivalence. No global model ranking or conclusion is selected from a single RMSE.
