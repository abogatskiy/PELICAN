# PELICAN Network for Particle Physics

    Permutation Equivariant, Lorentz Invariant/Covariant Aggregator Network for applications in Particle Physics.
    At the moment it includes two main variants: a classifier (for e.g. top-tagging) and a 4-momentum regression network. 

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* python >=3.9
* pytorch >=1.10
* h5py
* colorlog
* scikit-learn
* tensorboard (if using --summarize)
* optuna (if using the optuna script)
* psycopg2-binary (if using optuna with a distributed file sharing system)

### Installing

* No installatio required -- just use one of the top level scripts.

### General Usage Tips

* The classifier can be used via the script `train_pelican_classifier.py`. The 4-momehtum regressor is `train_pelican_cov.py`. 
    The main required argument is `--datadir` since it provides the datasets. 
* See data/sample_data/ for a small example of datasets. Each datapoint contains some number of input 4-momenta (E,p_x,p_y,p_z) under the key `Pmu`, 
    target 4-momenta (e.g. the true top momentum) under `truth_Pmu` and a classification label under `is_signal`. Regression datasets can contain multiple target 4-momenta under keys
    like `truth_Pmu_0` etc. The choice of the target key is controlled by the argument `--target`.
* The same script can be used for inference on the test dataset when it is run with the flag `--eval`. 
* Model checkpoints can be loaded via `--load` for inference or continued training. 
* By default the model is run on a GPU, but CPU evaluation can be forced via `--cpu`.
* The argument `--verbose` (correspodinginly `--no-verbose`) can be used to write per-minibatch stats into the log.
* The number of particles in each event can vary. Most computations are done with masks that properly exclude zero 4-momenta. The argument `--nobj` sets the maximum number of particles loaded from the dataset. Argument `--add-beams` also appends two "beam" particles of the form (1,0,0,±1) to the inputs to help the network learn that bias in the dataset due to the fixed z-axis. With this flag, the intermediate tensors in Eq2to2 layers will have shape [Batch, 2+Nobj, 2+Nobj, Channel].
* The network insludes learnable rescaling of certain tensors by exponents of the ratio `Nobj/Nobj_avg`, where `Nobj_avg` is a constant set by `--nobj-avg` that should indicate the typical (or average) number of particles in one event. In the large top-tagging dataset used in the paper this number of 49.
* The input 4-monenta can be uniformly scaled with a multiplicative `--scale` (default 1.0). The beams are not rescaled.
    In case of momentum regression, the 4-momentum output by the network is automatically divided by the same scale to better match the scale of the target.

### Outputs of the script

* Logfiles are kept in the `log/` folder. By default these contain initializtion information followed by training and validation stats for each epoch, and testing results at the end. The argument `--prefix` is used to name all files. Re-running the script without changing the prefix will overwrite all output files unless the flag `--load` is used.
* If `--summarize` is on, then Tensorboard summaries are saved into a folder with the same name as the log. If `--summarize-scv` is on, then per-minibatch stats are written into a separate CSV file.
* At the end of evaluation on the testing set, the stats are written into a CSV file whose name ends in `Best.metrics.csv` (for the model checkpoint with the best validation score) and `Final.metrics.csv` (for the model checkpoint from the last epoch). If there are multiple runs whose prefixes only differ by text after a dash (e.g. `run-1, run-2, run-3`, etc.) then their metrics will be appended to the same CSV.
* Model outputs (predictions) are saved as .pt files in `predict/`
* Model checkpoints are saved as .pt files in `model/`.

### Executing the training scripts

* Here is an example of a command that starts training the classifier on the sample dataset that is part of this repository. Optimally there should be three files with names train.h5, valid.h5, and test.h5.
```
python3 train_pelican_classifier.py --datadir=./data/sample_data/run12 --target=is_signal --nobj=85 --nobj-avg=40 --num-epoch=64 --num-train=60000 --num-valid=60000 --batch-size=64 --prefix=classifier --optim=adamw --activation=leakyrelu --factorize --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.025
```
* Similarly for regression:
```
python3 train_pelican_cov.py --datadir=./data/datadir --target=truth_Pmu_2 --nobj=85 --nobj-avg=20 --num-epoch=64 --num-train=60000 --num-valid=60000 --batch-size=64 --prefix=classifier --optim=adamw --activation=leakyrelu --factorize --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.025
```


## Authors

Contributors names and contact info

ex. Anonymous
ex. [@Anon](https://anon.anon/anon)


## Acknowledgments

Inspiration, code snippets, etc.
* [Masked BatchNorm](https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py)
* [Gradual Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py)
* [whichcraft](https://github.com/cookiecutter/whichcraft)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
