# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PELICAN: Permutation Equivariant, Lorentz Invariant/Covariant Aggregator Network (arXiv:2211.00454). A finished,
published model: pairwise dot products of the constituents -> input encoder -> rank-2 permutation-equivariant blocks ->
pooling. The package must not change what trained models compute.

## Commands

```bash
pip install -e ".[all]"                       # package + tensorboard, pytest (env: conda py310)
pytest                                        # synthetic data, ~10 s; MPS tests skip without MPS
pytest tests/test_models.py -k regression     # a subset
pelican train --datadir data/v0 --config config/toptag.yaml --config config/48k.yaml --prefix NAME [--mps|--cpu]
pelican eval runs/NAME                        # re-evaluate a run from its saved config.yaml
pelican info --config config/48k.yaml         # model + parameter count, no data
python scripts/convert_checkpoint.py model/NAME.pt   # old-script checkpoints -> runs/NAME/
```

## Layout

- `pelican/models/model.py`: `PelicanModel` (spurions -> pairwise invariants -> encoder -> lift of particle scalars -> `EqNet` -> message layer -> pooling `EqLayer` -> output MLP; regression output sum_i w_i p_i).
- `pelican/symmetry.py`: stabilizers (spurion 4-vectors), `pair_invariants` (dot products, stabilizer-specific products for `--method input`, IRC-safe angles and weights), `symmetry_vectors`, PELICAN's own `dot4`.
- `pelican/nn/`: `equivariant.py` (`EqLayer` for orders 1/2 -> 0/1/2 + `EqNet`, fast pair-output path), `encoders.py` (`InputEncoder`, `powerlog`), `layers.py` (`mlp`, `MessageNet`), `norms.py` (masked batchnorm).
- `pelican/softdrop.py`: Lorentz-invariant Soft Drop multiplicity, the normalization of IRC-safe mean aggregations.
- `pelican/legacy.py` + `scripts/convert_checkpoint.py`: conversion of checkpoints written by the pre-package scripts.
- Shared with CARDINAL (keep identical apart from package names): `pelican/data/`, `pelican/train/` (trainer, tasks, optim, ui, distributed, metrics), `pelican/lorentz.py`, `pelican/nn/{layers,norms}.py`, `pelican/cli.py` and `pelican/config.py` (except the model-specific parts), `pelican/selftest.py`, `scripts/export_predictions.py`. CARDINAL's optuna support (`search.py`, the `optuna` command, the trainer's trial hooks) was deliberately left out.

## Design points that span files

- **Spurions are particles** (unlike CARDINAL): `--method spurions` prepends the stabilizer's 4-vectors to every event with a one-hot type label, so `nobj`, the IRC weights and the regression sum include them. `--method input` instead adds stabilizer-invariant component products to the pairwise inputs (for `se2` these fix (1,0,0,1), not the spurion (1,0,0,-1); see `symmetry_vectors`).
- **Numerics are pinned to the pre-package code**: `symmetry.dot4` keeps PELICAN's formula 2 E1 E2 - sum p1 p2 (lorentz.dot4 rounds differently and the encoder amplifies it); `EqLayer` keeps PELICAN's conventions where they differ from CARDINAL's: the 1->2 and 2->1 layers scale all their ops by (nobj/nobj_avg)^alpha, IRC-weighted sums still divide by nobj_avg, IRC mode keeps all 15 ops and the diagonal bias, and only 2->2 has a diagonal bias.
- **IRC-safe mode** is exactly IR safe but only approximately collinear safe (diagonal ops, diagonal bias and the encoder see a collinear split); `tests/test_irc.py` documents this with a strict xfail.
- **dtypes:** everything, including the dot products, is computed in the model dtype; float32 results are sensitive to rounding for boosted, nearly collinear constituents. Nothing may depend on `torch.get_default_dtype()` (the tests set it to float64).
- **Run folders:** `runs/<prefix>/` with `config.yaml`, `train.log`, `metrics.csv`, `checkpoint.pt`, `best.pt`, `test_metrics.json`, `predictions/`. Training refuses to overwrite an existing checkpoint without `--load`.

## Known issues kept on purpose (fixing them changes trained models)

- Classification logits pass through the output activation by default (`--mlp-out-activate-last`; LeakyReLU shrinks negative logits 100x).
- `se2` regression with `--method input` is not Lorentz covariant (`_regression_prediction`; strict xfail in tests/test_models.py).

## Conventions

- Do not add Claude attribution (Co-Authored-By etc.) to commits or PRs.
- Commit messages: one sentence, optionally with one-sentence bullets for major file changes.
- Keep this file in sync with the code; it is tracked.
- The pre-package code (`src/`, `train_pelican_*.py`) lives in git history; checkpoints written by it convert with `scripts/convert_checkpoint.py`.
