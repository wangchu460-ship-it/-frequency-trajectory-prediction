# Physics-guided graph learning for device-level frequency trajectories

This is a dataset-excluded release assembled from the G0 mainline handoff.
It contains the canonical model/training source, frozen contracts, the formal
G0 checkpoint, selected run checkpoints, and audited aggregate results for the
IEEE 39-bus, NPCC 140-bus, and WECC 179-bus evaluations.

The release is intended for code inspection and result verification. The
simulation arrays and relation caches are not redistributed. Obtain the data
under the terms of their original sources and configure local paths before
running training or evaluation.

## Contents

```text
src/model/          G0 model, decoder, loss, and graph layers
src/training/       system-balanced training and metric helpers
src/selection/      Validation checkpoint-selection implementation
src/runtime/        relation/control adapters, locks, and initial seed
configs/            G0 training contract and frozen selection contracts
scripts/            held-out evaluation, audit, and reporting utilities
checkpoints/        G0_BALANCED_e53 and archived comparison checkpoints
results/training/   run inventory, configurations, and training histories
results/validation/ selection and matched-comparison evidence
results/heldout_aggregate/
                    table-ready aggregate results and audit records
provenance/         protocol locks, manifests, and handoff records
```

## Model and information boundary

The model predicts SG, GFM, and GFL device-frequency trajectories. Its
post-contingency device-level electrical relation `F` is constructed from the
specified network and operating-point information available at prediction
initialization. Four 128-dimensional graph blocks use the same relation for
F-message aggregation and F-Laplacian differential propagation. A direct
time-conditioned decoder queries each time point independently. Bus records
remain input features/readout context, but bus nodes are not endpoints of
F propagation and no bus-message branch is used.

The relation is not a dynamic simulator and is not updated with the target
trajectory. The exact semantics are specified in `configs/` and the source
files; historical names retained in provenance are not alternative model
definitions.

## Frozen formal checkpoint

`checkpoints/G0_BALANCED_e53.pt` is the formal seed-789 checkpoint. Its
SHA-256 is:

```text
4c61eee32611eb8f63c5efbef92b2821482fd4c2ee54583ef88dcb3195d00a2a
```

The corresponding contract is
`configs/G0_BALANCED_e53_config.json`. It records the batch contract,
optimizer, normalization, loss coefficient, relation protocol, and data
read counters used for selection.

## Training

The core training entry point is `src/training/balanced_train.py`. It expects
local prepared inputs and relation/control artifacts; no absolute Windows or
server path is required by the release layout once those paths are supplied
through the local configuration. A compatible environment is required:

```text
Python 3.10 or 3.11
PyTorch 2.x with CUDA (GPU is recommended)
torch-geometric 2.x
NumPy, SciPy, and pandas
```

The original runs used FP32, AdamW, learning rate `1e-4`, weight decay
`1e-4`, gradient-norm clipping `1.0`, and strict four-IEEE39 plus four-NPCC140
system-level sampling. Run the supplied audit/preflight utilities before a
long training job. Do not use Test, OOD, or WECC data for training or
checkpoint selection.

## Evaluation and results

`scripts/` contains the frozen evaluation/audit utilities. The aggregate
files in `results/heldout_aggregate/` preserve the published metric values,
paired comparisons, population counts, and provenance without including raw
prediction arrays. They are descriptive result artifacts; recomputation of
metrics requires the corresponding local reference data and predictions.
The path assumptions inherited by the audit scripts are documented in
`DATA_PATHS.md` and must be replaced on a new machine.

The release does not silently claim universal transfer. The WECC results are
reported as a Generalization-set evaluation with frozen source parameters and
target-system relation construction.

## Data exclusion

This release contains no simulation dataset arrays. In particular, do not
copy the original `UNIFIED_F_FCAR_FULL_V1_PACKAGE.zip` into this repository:
that historical package contains prepared Train/Validation arrays and is not
part of this dataset-excluded release. Relation caches and manifests must be
obtained separately and pass their recorded integrity checks.

## Reproducibility notes

- Verify all checkpoint hashes before loading a model.
- Use the supplied contracts rather than inferring settings from filenames.
- Preserve the stated Validation/Test/Generalization split and aggregation
  hierarchy.
- Record the data manifest, software versions, device, seed, and command line
  for every new run.
- Large checkpoints should be distributed through Git LFS or GitHub Release
  assets rather than ordinary Git history.

This repository is released for research use. See `LICENSE` and
`CITATION.cff` for attribution and reuse information.
