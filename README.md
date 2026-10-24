# PELICAN

PELICAN (Permutation Equivariant, Lorentz Invariant/Covariant Aggregator Network) is a network for particle physics
that takes 4-momentum inputs (e.g. jet constituents), builds Lorentz-invariant features from all pairwise dot products,
and processes them with rank-2 permutation-equivariant layers. It solves tasks such as jet tagging and 4-momentum
regression in an exactly Lorentz-invariant (or covariant) way. Embedding the symmetries reduces the model size while
delivering state-of-the-art performance, and improves sample efficiency.

arXiv: https://arxiv.org/abs/2211.00454 (see also CARDINAL, its successor, which shares this package's infrastructure
and command line).

PELICAN supports classification (`--task classification`, e.g. top tagging) and Lorentz-covariant 4-vector regression
(`--task regression`), where the prediction is Σᵢ wᵢ pᵢ with learned invariant weights wᵢ.

## Installation

```bash
pip install -e .                 # core: torch, numpy, h5py, scikit-learn, pyyaml, rich
pip install -e ".[all]"          # + tensorboard, pytest
```

This installs the `pelican` command. `python -m pelican` works the same way.

## Quick start

```bash
# the sample data in this repository (data/sample_data/run12 holds train.h5, valid.h5, test.h5)
pelican train --datadir data/sample_data/run12 --config config/48k.yaml --batch-size 64 --prefix classifier

# top tagging (data/v0 holds out_train.h5, out_valid.h5, out_test.h5)
pelican train --datadir data/v0 --config config/toptag.yaml --config config/48k.yaml --prefix toptag-1

# evaluate a finished run again (e.g. on another test file)
pelican eval runs/toptag-1 --testfile other_test.h5

# 4-momentum regression
pelican train --task regression --target truth_Pmu --datadir data/sample_data/run12 --nobj 80 --nobj-avg 20 --prefix reg

# print a model and its parameter count without data
pelican info --config config/48k.yaml
```

`pelican <command> --help` lists every option. Options can also come from YAML files given with `--config`
(repeatable; later files override earlier ones, and command-line flags override all of them). YAML keys are the
option names with underscores, e.g. `num_channels_m: [[60], [60]]`. Unknown keys are an error, and keys of the
pre-package scripts (`config`, `num_channels_2to2`, `lr_decay_type`, ...) are translated with a warning.

On a terminal, training shows a live progress bar and one summary line per epoch. Under slurm (no terminal) it
prints plain timestamped lines. `--log-every N` adds a line every N minibatches.

## Data

`--datadir` must contain HDF5 files whose names include `train`, `valid` and `test`; a split may span several
files. Each file holds equal-length arrays with one row per event:

* `Pmu`: constituent 4-momenta (E, px, py, pz), zero-padded to a fixed number of slots. Padding is detected as E = 0.
* the target named by `--target`: `is_signal` (default), class indices or one-hot vectors (`--num-classes > 2`), or
  4-vectors such as `truth_Pmu` for regression.
* any other per-event or per-particle arrays, which are passed through to the model.

`--nobj` keeps the first N constituents of each event; `--nobj-avg` should be the typical multiplicity (49 for top
tagging). `--dataset jc` adds the JetClass per-particle features (charge, particle-ID flags, impact parameters).
`data/sample_data/run12` is a small sample: 50% top-quark jets and 50% background jets, with `truth_Pmu` = 0 for the
background, so it is not a meaningful regression benchmark (the resolution metrics skip events with a zero target).

## Model

The inputs are the dot products pᵢ·pⱼ of all pairs of constituents. A linear map and learnable power functions
(xᵃ − 1)/a (a → 0 gives log) embed them into channels; per-particle features (spurion labels, `--dataset` features)
are lifted to pairs by a rank-1 → rank-2 equivariant layer and appended. A stack of rank-2 blocks follows, each a
pairwise MessageNet and a permutation-equivariant 2 → 2 layer with the 15-element basis of Maron et al.; then a final
message layer, pooling to rank 0 (classification) or rank 1 (regression weights), and an output MLP.

Widths: `--num-channels-m` (MessageNet hidden widths per block), `--num-channels-eq` (input width of each 2 → 2 layer),
`--num-channels-m-out`, `--num-channels-out`, `--num-channels-scalar` (pair channels from the per-particle features).
`config/{3k,6k,11k,24k,48k,208k,JCxl}.yaml` are the model sizes used in the paper and later work.

Aggregations (`--agg` in the blocks, `--agg-out` for pooling and lifting) are a single letter: `s` sum (divided by
`--nobj-avg`), `m` mean, `x` max, `n` min; upper case adds learnable (nobj/nobj_avg)^α scalings.

## Symmetry

`--stabilizer` selects the residual symmetry group, and `--method` how it is broken:

* `spurions` (default): fixed 4-vectors (beam directions, lab axes) are added to every event as extra particles with a
  one-hot type label, so all inputs remain dot products.
* `input`: no extra particles; the pairwise inputs also include products of the components the stabilizer leaves
  invariant (e.g. E and p_z for `so2`).

| stabilizer | spurions | invariant under |
|---|---|---|
| `so13` | none | full Lorentz group |
| `so3`, `so12`, `se2` | one | rotations / SO(1,2) about the beam / ISO(2) |
| `so2` (default), `so2_0` | two beams | rotations about the beam axis |
| `R` | x and y axes | boosts along the beam |
| `1`, `1_0`, `11`, `11_0` | all axes | nothing (lab frame); with `--method input`, `11*` also use the dot products |

With `--method input`, `se2` uses E − p_z and is invariant under the ISO(2) fixing (1,0,0,1), while its spurion is
(1,0,0,−1).

**IRC safety** (`--irc-safe`): the inputs become jet-frame angles 2(1 − cos θᵢⱼ), and every aggregation is weighted by
the constituents' energy fractions. The model is then exactly IR safe (zero-momentum particles have no effect) but only
approximately collinear safe. With a mean aggregation, the particle count is replaced by a Lorentz-invariant Soft Drop
multiplicity (slow).

## Outputs

Each run writes to `runs/<prefix>/`:

| file | contents |
|---|---|
| `config.yaml` | the full resolved configuration (`pelican eval` reads it back) |
| `train.log` | log of the run |
| `metrics.csv` | per-epoch train and validation metrics |
| `checkpoint.pt`, `best.pt` | last epoch (resume with `--load`) and lowest validation loss |
| `test_metrics.json` | test metrics of the best and final models |
| `predictions/` | `<best\|final>_test.pt` (predictions and targets) and ROC curves as CSV |

Runs whose prefixes differ only after a dash (`toptag-1`, `toptag-2`, ...) also append their test metrics to
`runs/toptag.summary.csv`. `--summarize` writes TensorBoard logs to `runs/<prefix>/tensorboard/`. Training refuses to
overwrite a run that already has a checkpoint; use `--load` to resume it or pick a new `--prefix`.

Checkpoints of the pre-package scripts (`model/<name>.pt`, `model/<name>_best.pt`) can be converted into a run folder
with `python scripts/convert_checkpoint.py model/<name>.pt`, then evaluated with `pelican eval runs/<name>`.

## Clusters and multiple GPUs

`scripts/export_predictions.py runs/<prefix>` converts a run's saved predictions to HDF5.

Multi-GPU training uses `torchrun --nproc_per_node=4 -m pelican train ...`. Example slurm jobs are in `slurm/`; on a
cluster, install the package into the job's environment once with `pip install -e .`.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The tests use synthetic jets, so they need no data. They check:

* invariance under random elements of each stabilizer's subgroup (covariance for regression), for both methods;
* permutation invariance, padding and batch independence;
* IR safety of IRC-safe models under zero-momentum insertion (collinear safety is only approximate);
* layer-level behaviour;
* CPU vs MPS agreement, when MPS is available;
* end-to-end CLI runs, and conversion of checkpoints of the pre-package scripts.

## Authors

Alexander Bogatskiy, Flatiron Institute

Jan T. Offermann, University of Chicago

Timothy Hoffman, University of Chicago

Xiaoyang Liu, University of Chicago

## Acknowledgments

* [Masked BatchNorm](https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py)
* Permutation-equivariant layers after [permeqlayers](https://github.com/horacepan/permeqlayers)

## License

This project is licensed under the MIT License; see LICENSE.md for details.
