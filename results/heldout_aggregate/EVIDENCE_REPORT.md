# V2 frozen evaluation evidence

All 39 inference passes and Validation/array pairing checks completed. Results are reported by scientific endpoint, without a single-score ranking.

## Frozen checkpoints

| Model | Epoch | Boundary status |
|---|---:|---|
| G0_BALANCED | 53 | NOT_AT_BOUNDARY |
| A1_NO_BALANCE | 53 | NOT_AT_BOUNDARY |
| A2_NO_SPATIAL | 32 | NOT_AT_BOUNDARY |
| A3_F_ONLY | 44 | NOT_AT_BOUNDARY |
| A4_L_ONLY | 26 | NOT_AT_BOUNDARY |
| A5_NO_PHYSICAL | 47 | NOT_AT_BOUNDARY |
| A6_PRE_EVENT | 56 | TRAINING_BOUNDARY_NOT_RESOLVED |
| E1_GRAPHICAL_DEEPONET | 57 | PLATEAU/OSCILLATION |
| E2_GAT_H128 | 52 | NOT_AT_BOUNDARY |
| E3_GCN_H128 | 58 | TRAINING_BOUNDARY_NOT_RESOLVED |
| E4_UGCN | 54 | NOT_AT_BOUNDARY |
| E5_GAT_H384 | 55 | NOT_AT_BOUNDARY |
| E6_GCN_H384 | 55 | NOT_AT_BOUNDARY |

## Core Test and transfer comparison

Each difference is model minus G0. Negative indicates lower error. Confidence intervals are nominal group-bootstrap intervals; “unresolved” is not equivalence. Source HIGH results and WECC ordinary usable results are kept distinct.

| Dataset | System | Model | Metric | Window | Group macro | Δ vs G0 [95% CI] | Group win |
|---|---|---|---|---|---:|---|---:|
| test | NPCC140 | G0_BALANCED | differential_rmse | 0_2 | 0.0571745 | reference | — |
| test | NPCC140 | G0_BALANCED | differential_rmse | 2_10 | 0.176062 | reference | — |
| test | IEEE39 | G0_BALANCED | differential_rmse | 0_2 | 0.297039 | reference | — |
| test | IEEE39 | G0_BALANCED | differential_rmse | 2_10 | 0.436126 | reference | — |
| wecc1600 | WECC179 | G0_BALANCED | trajectory_rmse | 0_2 | 0.104837 | reference | — |
| wecc1600 | WECC179 | G0_BALANCED | differential_rmse | 0_2 | 0.069406 | reference | — |
| wecc1600 | WECC179 | G0_BALANCED | trajectory_rmse | 2_10 | 0.131988 | reference | — |
| wecc1600 | WECC179 | G0_BALANCED | differential_rmse | 2_10 | 0.0836594 | reference | — |
| test | NPCC140 | A1_NO_BALANCE | differential_rmse | 0_2 | 0.0573555 | 0.000181019 [-0.00152281, 0.00183144] | 48.7% |
| test | NPCC140 | A1_NO_BALANCE | differential_rmse | 2_10 | 0.175204 | -0.000858787 [-0.00693904, 0.00509388] | 45.5% |
| test | IEEE39 | A1_NO_BALANCE | differential_rmse | 0_2 | 0.286578 | -0.0104608 [-0.0209014, 0.000843261] | 61.5% |
| test | IEEE39 | A1_NO_BALANCE | differential_rmse | 2_10 | 0.429904 | -0.00622198 [-0.018842, 0.00612068] | 51.3% |
| wecc1600 | WECC179 | A1_NO_BALANCE | trajectory_rmse | 0_2 | 0.0993894 | -0.00544754 [-0.00719105, -0.0036546] | 55.7% |
| wecc1600 | WECC179 | A1_NO_BALANCE | differential_rmse | 0_2 | 0.0678205 | -0.00158549 [-0.00242356, -0.000747733] | 56.1% |
| wecc1600 | WECC179 | A1_NO_BALANCE | trajectory_rmse | 2_10 | 0.0954143 | -0.0365733 [-0.0392158, -0.0339854] | 79.2% |
| wecc1600 | WECC179 | A1_NO_BALANCE | differential_rmse | 2_10 | 0.0574979 | -0.0261616 [-0.0297472, -0.0226176] | 61.4% |
| test | NPCC140 | A2_NO_SPATIAL | differential_rmse | 0_2 | 0.058318 | 0.0011435 [-0.00102598, 0.00325365] | 61.9% |
| test | NPCC140 | A2_NO_SPATIAL | differential_rmse | 2_10 | 0.185169 | 0.00910686 [0.00301758, 0.0156277] | 46.0% |
| test | IEEE39 | A2_NO_SPATIAL | differential_rmse | 0_2 | 0.310831 | 0.0137921 [0.00758465, 0.0204217] | 36.7% |
| test | IEEE39 | A2_NO_SPATIAL | differential_rmse | 2_10 | 0.452894 | 0.0167677 [0.00133482, 0.0322376] | 49.1% |
| wecc1600 | WECC179 | A2_NO_SPATIAL | trajectory_rmse | 0_2 | 0.0617167 | -0.0431203 [-0.0452216, -0.0409155] | 94.9% |
| wecc1600 | WECC179 | A2_NO_SPATIAL | differential_rmse | 0_2 | 0.0400772 | -0.0293288 [-0.0305362, -0.0281374] | 94.2% |
| wecc1600 | WECC179 | A2_NO_SPATIAL | trajectory_rmse | 2_10 | 0.0947371 | -0.0372505 [-0.0393135, -0.0351706] | 87.8% |
| wecc1600 | WECC179 | A2_NO_SPATIAL | differential_rmse | 2_10 | 0.0723409 | -0.0113185 [-0.0135133, -0.0090586] | 80.8% |
| test | NPCC140 | A3_F_ONLY | differential_rmse | 0_2 | 0.0563364 | -0.000838074 [-0.00255315, 0.00082562] | 53.4% |
| test | NPCC140 | A3_F_ONLY | differential_rmse | 2_10 | 0.181016 | 0.00495322 [-4.55852e-05, 0.00975151] | 42.3% |
| test | IEEE39 | A3_F_ONLY | differential_rmse | 0_2 | 0.299132 | 0.00209314 [-0.0043539, 0.00844671] | 51.3% |
| test | IEEE39 | A3_F_ONLY | differential_rmse | 2_10 | 0.450896 | 0.0147699 [-0.000741034, 0.0288264] | 47.3% |
| wecc1600 | WECC179 | A3_F_ONLY | trajectory_rmse | 0_2 | 0.104035 | -0.000802405 [-0.00203833, 0.000423246] | 47.0% |
| wecc1600 | WECC179 | A3_F_ONLY | differential_rmse | 0_2 | 0.0697681 | 0.000362123 [-0.000684466, 0.00134598] | 38.6% |
| wecc1600 | WECC179 | A3_F_ONLY | trajectory_rmse | 2_10 | 0.115144 | -0.0168435 [-0.0195639, -0.0141404] | 59.5% |
| wecc1600 | WECC179 | A3_F_ONLY | differential_rmse | 2_10 | 0.0470658 | -0.0365936 [-0.0408489, -0.0322793] | 55.8% |
| test | NPCC140 | A4_L_ONLY | differential_rmse | 0_2 | 0.0627467 | 0.00557228 [0.00302402, 0.00843862] | 50.3% |
| test | NPCC140 | A4_L_ONLY | differential_rmse | 2_10 | 0.19196 | 0.0158981 [0.00856828, 0.023903] | 39.7% |
| test | IEEE39 | A4_L_ONLY | differential_rmse | 0_2 | 0.302799 | 0.00576079 [-0.000713101, 0.0118723] | 41.6% |
| test | IEEE39 | A4_L_ONLY | differential_rmse | 2_10 | 0.452321 | 0.0161954 [0.000299876, 0.0315638] | 45.1% |
| wecc1600 | WECC179 | A4_L_ONLY | trajectory_rmse | 0_2 | 0.0738836 | -0.0309534 [-0.0332008, -0.0287068] | 82.2% |
| wecc1600 | WECC179 | A4_L_ONLY | differential_rmse | 0_2 | 0.0487499 | -0.0206561 [-0.0219466, -0.0194348] | 89.6% |
| wecc1600 | WECC179 | A4_L_ONLY | trajectory_rmse | 2_10 | 0.0651463 | -0.0668413 [-0.0700685, -0.0636244] | 93.8% |
| wecc1600 | WECC179 | A4_L_ONLY | differential_rmse | 2_10 | 0.0325648 | -0.0510946 [-0.0564755, -0.0458008] | 73.4% |
| test | NPCC140 | A5_NO_PHYSICAL | differential_rmse | 0_2 | 0.0625668 | 0.00539235 [0.00325497, 0.00752436] | 32.8% |
| test | NPCC140 | A5_NO_PHYSICAL | differential_rmse | 2_10 | 0.187929 | 0.0118664 [0.00458744, 0.0193808] | 44.4% |
| test | IEEE39 | A5_NO_PHYSICAL | differential_rmse | 0_2 | 0.30191 | 0.00487184 [-0.00333829, 0.0128921] | 50.0% |
| test | IEEE39 | A5_NO_PHYSICAL | differential_rmse | 2_10 | 0.450817 | 0.0146915 [-0.010009, 0.0388148] | 53.1% |
| wecc1600 | WECC179 | A5_NO_PHYSICAL | trajectory_rmse | 0_2 | 0.0910568 | -0.0137802 [-0.0162277, -0.0113026] | 68.5% |
| wecc1600 | WECC179 | A5_NO_PHYSICAL | differential_rmse | 0_2 | 0.0637658 | -0.00564023 [-0.00707877, -0.00424251] | 54.5% |
| wecc1600 | WECC179 | A5_NO_PHYSICAL | trajectory_rmse | 2_10 | 0.0724477 | -0.0595399 [-0.0625699, -0.056521] | 89.9% |
| wecc1600 | WECC179 | A5_NO_PHYSICAL | differential_rmse | 2_10 | 0.0377333 | -0.0459262 [-0.0504323, -0.0414983] | 81.5% |
| test | NPCC140 | A6_PRE_EVENT | differential_rmse | 0_2 | 0.0573277 | 0.000153288 [-0.00140242, 0.00153077] | 46.0% |
| test | NPCC140 | A6_PRE_EVENT | differential_rmse | 2_10 | 0.177358 | 0.00129532 [-0.00196298, 0.00443893] | 48.7% |
| test | IEEE39 | A6_PRE_EVENT | differential_rmse | 0_2 | 0.297933 | 0.00089415 [-0.00396668, 0.00557976] | 48.2% |
| test | IEEE39 | A6_PRE_EVENT | differential_rmse | 2_10 | 0.439392 | 0.00326656 [-0.00798882, 0.0143903] | 48.7% |
| wecc1600 | WECC179 | A6_PRE_EVENT | trajectory_rmse | 0_2 | 0.0884922 | -0.0163447 [-0.0177609, -0.01488] | 74.8% |
| wecc1600 | WECC179 | A6_PRE_EVENT | differential_rmse | 0_2 | 0.0559017 | -0.0135043 [-0.0145477, -0.0125013] | 79.6% |
| wecc1600 | WECC179 | A6_PRE_EVENT | trajectory_rmse | 2_10 | 0.11452 | -0.017468 [-0.0198468, -0.0152061] | 63.3% |
| wecc1600 | WECC179 | A6_PRE_EVENT | differential_rmse | 2_10 | 0.046155 | -0.0375044 [-0.0408014, -0.0342863] | 85.8% |
| test | NPCC140 | E1_GRAPHICAL_DEEPONET | differential_rmse | 0_2 | 0.0597666 | 0.00259218 [-0.000510412, 0.00595047] | 31.2% |
| test | NPCC140 | E1_GRAPHICAL_DEEPONET | differential_rmse | 2_10 | 0.196018 | 0.0199561 [0.0121349, 0.0282087] | 30.7% |
| test | IEEE39 | E1_GRAPHICAL_DEEPONET | differential_rmse | 0_2 | 0.304945 | 0.00790624 [-0.00257893, 0.0178679] | 35.8% |
| test | IEEE39 | E1_GRAPHICAL_DEEPONET | differential_rmse | 2_10 | 0.481025 | 0.0448996 [0.0136417, 0.0759848] | 36.7% |
| wecc1600 | WECC179 | E1_GRAPHICAL_DEEPONET | trajectory_rmse | 0_2 | 0.104515 | -0.000322253 [-0.00350983, 0.00295587] | 56.4% |
| wecc1600 | WECC179 | E1_GRAPHICAL_DEEPONET | differential_rmse | 0_2 | 0.10355 | 0.0341441 [0.0307547, 0.0377373] | 34.5% |
| wecc1600 | WECC179 | E1_GRAPHICAL_DEEPONET | trajectory_rmse | 2_10 | 0.322017 | 0.19003 [0.181075, 0.199362] | 10.1% |
| wecc1600 | WECC179 | E1_GRAPHICAL_DEEPONET | differential_rmse | 2_10 | 0.370216 | 0.286556 [0.278669, 0.294285] | 2.1% |
| test | NPCC140 | E2_GAT_H128 | differential_rmse | 0_2 | 0.0567658 | -0.000408625 [-0.00223212, 0.00131219] | 57.7% |
| test | NPCC140 | E2_GAT_H128 | differential_rmse | 2_10 | 0.181969 | 0.00590705 [-0.00116317, 0.0132102] | 50.8% |
| test | IEEE39 | E2_GAT_H128 | differential_rmse | 0_2 | 0.30686 | 0.00982116 [0.000845026, 0.0189202] | 43.4% |
| test | IEEE39 | E2_GAT_H128 | differential_rmse | 2_10 | 0.487148 | 0.0510224 [0.0299166, 0.0725027] | 38.1% |
| wecc1600 | WECC179 | E2_GAT_H128 | trajectory_rmse | 0_2 | 0.0691932 | -0.0356437 [-0.0378341, -0.0334503] | 81.8% |
| wecc1600 | WECC179 | E2_GAT_H128 | differential_rmse | 0_2 | 0.0462203 | -0.0231857 [-0.0248393, -0.0215669] | 84.0% |
| wecc1600 | WECC179 | E2_GAT_H128 | trajectory_rmse | 2_10 | 0.0894339 | -0.0425537 [-0.0457284, -0.0393931] | 80.1% |
| wecc1600 | WECC179 | E2_GAT_H128 | differential_rmse | 2_10 | 0.0388696 | -0.0447898 [-0.0493076, -0.0403947] | 84.8% |
| test | NPCC140 | E3_GCN_H128 | differential_rmse | 0_2 | 0.0567688 | -0.000405671 [-0.0024044, 0.0014399] | 42.9% |
| test | NPCC140 | E3_GCN_H128 | differential_rmse | 2_10 | 0.18267 | 0.00660775 [-0.000319494, 0.0136845] | 40.2% |
| test | IEEE39 | E3_GCN_H128 | differential_rmse | 0_2 | 0.302514 | 0.00547499 [-0.00294506, 0.0139327] | 42.5% |
| test | IEEE39 | E3_GCN_H128 | differential_rmse | 2_10 | 0.468586 | 0.0324598 [0.00815057, 0.0562166] | 52.7% |
| wecc1600 | WECC179 | E3_GCN_H128 | trajectory_rmse | 0_2 | 0.0816037 | -0.0232332 [-0.0258188, -0.0205861] | 67.3% |
| wecc1600 | WECC179 | E3_GCN_H128 | differential_rmse | 0_2 | 0.0543066 | -0.0150994 [-0.0168382, -0.0133498] | 65.5% |
| wecc1600 | WECC179 | E3_GCN_H128 | trajectory_rmse | 2_10 | 0.108443 | -0.0235445 [-0.0269261, -0.0200997] | 68.0% |
| wecc1600 | WECC179 | E3_GCN_H128 | differential_rmse | 2_10 | 0.0654266 | -0.0182329 [-0.0232417, -0.0131695] | 65.9% |
| test | NPCC140 | E4_UGCN | differential_rmse | 0_2 | 0.057876 | 0.000701539 [-0.00141173, 0.00289584] | 30.2% |
| test | NPCC140 | E4_UGCN | differential_rmse | 2_10 | 0.20023 | 0.0241681 [0.0149202, 0.0339073] | 32.3% |
| test | IEEE39 | E4_UGCN | differential_rmse | 0_2 | 0.312077 | 0.0150385 [-0.0016372, 0.0354389] | 44.2% |
| test | IEEE39 | E4_UGCN | differential_rmse | 2_10 | 0.496758 | 0.0606322 [0.0221887, 0.100965] | 43.8% |
| wecc1600 | WECC179 | E4_UGCN | trajectory_rmse | 0_2 | 0.0488522 | -0.0559848 [-0.0581675, -0.0537668] | 96.2% |
| wecc1600 | WECC179 | E4_UGCN | differential_rmse | 0_2 | 0.0352916 | -0.0341144 [-0.0356318, -0.0326435] | 97.2% |
| wecc1600 | WECC179 | E4_UGCN | trajectory_rmse | 2_10 | 0.0643112 | -0.0676763 [-0.0716933, -0.0636599] | 86.4% |
| wecc1600 | WECC179 | E4_UGCN | differential_rmse | 2_10 | 0.0642508 | -0.0194086 [-0.0250424, -0.0137971] | 48.4% |
| test | NPCC140 | E5_GAT_H384 | differential_rmse | 0_2 | 0.0535204 | -0.00365406 [-0.00583736, -0.00162046] | 62.4% |
| test | NPCC140 | E5_GAT_H384 | differential_rmse | 2_10 | 0.174977 | -0.00108488 [-0.00940352, 0.00716514] | 59.3% |
| test | IEEE39 | E5_GAT_H384 | differential_rmse | 0_2 | 0.295763 | -0.00127611 [-0.00970162, 0.0068209] | 55.8% |
| test | IEEE39 | E5_GAT_H384 | differential_rmse | 2_10 | 0.475143 | 0.039017 [0.0170724, 0.0611401] | 44.7% |
| wecc1600 | WECC179 | E5_GAT_H384 | trajectory_rmse | 0_2 | 0.0719944 | -0.0328425 [-0.0349144, -0.0306813] | 86.8% |
| wecc1600 | WECC179 | E5_GAT_H384 | differential_rmse | 0_2 | 0.048466 | -0.02094 [-0.0223971, -0.019497] | 91.7% |
| wecc1600 | WECC179 | E5_GAT_H384 | trajectory_rmse | 2_10 | 0.0785148 | -0.0534728 [-0.0564242, -0.0506054] | 92.3% |
| wecc1600 | WECC179 | E5_GAT_H384 | differential_rmse | 2_10 | 0.0297543 | -0.0539051 [-0.0591121, -0.0488331] | 83.7% |
| test | NPCC140 | E6_GCN_H384 | differential_rmse | 0_2 | 0.0555543 | -0.00162012 [-0.00382872, 0.000457399] | 41.8% |
| test | NPCC140 | E6_GCN_H384 | differential_rmse | 2_10 | 0.17988 | 0.0038177 [-0.00580435, 0.0130531] | 48.1% |
| test | IEEE39 | E6_GCN_H384 | differential_rmse | 0_2 | 0.305228 | 0.0081889 [-0.000354493, 0.0166376] | 48.2% |
| test | IEEE39 | E6_GCN_H384 | differential_rmse | 2_10 | 0.47663 | 0.040504 [0.0156282, 0.0657615] | 50.4% |
| wecc1600 | WECC179 | E6_GCN_H384 | trajectory_rmse | 0_2 | 0.0967489 | -0.008088 [-0.010294, -0.00579979] | 63.3% |
| wecc1600 | WECC179 | E6_GCN_H384 | differential_rmse | 0_2 | 0.0590272 | -0.0103788 [-0.0116959, -0.00903554] | 73.5% |
| wecc1600 | WECC179 | E6_GCN_H384 | trajectory_rmse | 2_10 | 0.129067 | -0.00292079 [-0.00599287, 0.000205383] | 49.2% |
| wecc1600 | WECC179 | E6_GCN_H384 | differential_rmse | 2_10 | 0.0587197 | -0.0249398 [-0.0299175, -0.0199977] | 58.1% |

## A5/G0 transfer boundary

Under the common V2 population and evaluator, A5 has higher Test NPCC140 HIGH differential error in both 0–2 and 2–10 s, with nominal group-bootstrap intervals above zero. IEEE39 corresponding intervals span zero. On WECC ordinary usable1402, A5 has lower trajectory and differential error in all four windows, with intervals below zero. This is an observed transfer limitation of the frozen G0 relative to its physical-path ablation. It does not establish that normalization is the cause, or that A5 is universally superior across source systems and endpoints.

## Claim boundaries

- A3/A4/A5 isolate message, differential and combined physical-path contributions; A5 is the unchanged model with both physical operators disabled, not a new device-only baseline.
- A2 isolates the spatial objective. Use its HIGH and all-device endpoints together.
- A1 isolates per-step system balancing. Preserve separate IEEE39/NPCC140 results.
- A6 compares PRE-event relations; its e56 training boundary remains unresolved.
- E1–E6 are the task-adapted baseline implementations. These results do not establish equivalence to every published implementation.
- WECC provides transfer evidence for the frozen models. Its risk-only cases and ordinary usable cases cannot be merged to claim ordinary prediction performance.
- Single-seed group bootstrap does not establish robustness over training initialization. No model modification or reselection is based on these results.

Full trajectory, all time windows, event strata, device families, cross-family pairs, source HIGH, label sensitivity, spatial amplitude, ranking/top-k and nadir/RoCoF are available in the matched CSV tables.

V1_V2_CONSISTENCY.json compares unchanged-checkpoint numerical outputs. Changed checkpoint results have no same-weight comparison and are explicitly marked. Changes caused by D3/risk population definitions are not numerical inference errors.
