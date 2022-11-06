# PELICAN Network for Particle Physics

    Permutation Equivariant, Lorentz Invariant/Covariant Aggregator Network for applications in Particle Physics.
    At the moment it includes two main variants: a classifier (for e.g. top-tagging) and a 4-momentum regression network. 

arXiv link: https://arxiv.org/abs/2211.00454

arXiv link: https://arxiv.org/abs/2211.00454

## Description

PELICAN is a network that takes 4-momentum inputs (e.g. jet constituents) and uses Lorentz-invariant features and permutation-equivariant layers to solve tasks such as classification and momentum regression in an exactly Lorentz-covariant way. Embedding the symmetries into the architecture drastically reduces the model size while still delivering state-of-the-art performance. Moreover, by using only physically/geometrically meaningful mathematical operations, PELICAN can provide viable physics models as opposed to black-box ML approaches that do not respect fundamental symmetries. Finally, equivariance massively improves sample efficiency (and generalizability from low amounts of training data). This repository contains two main top-level scripts, one for classification, and one for 4-momentum regression (currently outputting just one 4-vector prediction). These can be executed directly with no installation. The data folder contains a small sample dataset consisting of 50% top quark jets and 50% background jets (with the key `is_signal` identifying which is which).

## Getting Started

### Dependencies

* python >=3.9
* pytorch >=1.10
* h5py
* colorlog
* scikit-learn
* pyyaml
* tensorboard (if using --summarize)
* optuna (if using the optuna script)
* psycopg2-binary (if using optuna with a distributed file sharing system)

### Installing

* No installation required -- just use one of the top level scripts.

### General Usage Tips

* The classifier can be used via the script `train_pelican_classifier.py`. The 4-momentum regressor is `train_pelican_cov.py`. 
    The main required argument is `--datadir` since it provides the datasets. 
* See data/sample_data/ for a small example of datasets. Each datapoint contains some number of input 4-momenta (E,p_x,p_y,p_z) under the key `Pmu`, 
    target 4-momenta (e.g. the true top momentum) under `truth_Pmu` and a classification label under `is_signal`. Regression datasets can contain multiple target 4-momenta under keys like `truth_Pmu_0` etc. The choice of the target key is controlled by the argument `--target`.
* For classification, the number of target classes is set by the argument `--num-classes`. 
* When num_classes>2, the target is expected to be a one-hot encoding as opposed to a single integer label. Note that the way multi-class metrics are reported is also a bit different.
* The same scripts can be used for inference on the test dataset when it is run with the flag `--task=eval`. 
* Model checkpoints can be loaded via `--load` for inference or continued training (as long as the prefix is the same).
* By default the model is run on a GPU (`--cuda`), but CPU evaluation can be forced via `--cpu`.
* The argument `--verbose` (correspondingly `--no-verbose`) can be used to write per-minibatch stats into the log.
* The number of particles in each event can vary. Most computations are done with masks that properly exclude zero 4-momenta. The argument `--nobj` sets the maximum number of particles loaded from the dataset. Argument `--add-beams` also appends two "beam" particles of the form (1,0,0,±1) to the inputs to help the network learn that bias in the dataset due to the fixed z-axis. With this flag, the intermediate tensors in Eq2to2 layers will have shape [Batch, 2+Nobj, 2+Nobj, Channel].
* The network includes learnable rescaling of certain tensors by powers of the ratio `Nobj/Nobj_avg`, where `Nobj_avg` is a constant set by `--nobj-avg` that should indicate the typical (or average) number of particles in one event. In the large top-tagging dataset used in the paper this number of 49.
* The input 4-monenta can be uniformly scaled with a multiplicative `--scale` (default 1.0). The beams are not rescaled.
    In case of momentum regression, the 4-momentum output by the network is automatically divided by the same scale to better match the scale of the target.

### Outputs of the script

* Logfiles are kept in the `log/` folder. By default these contain initialization information followed by training and validation stats for each epoch, and testing results at the end. The argument `--prefix` is used to name all files. Re-running the script without changing the prefix will overwrite all output files unless the flag `--load` is used.
* If `--summarize` is on, then Tensorboard summaries are saved into a folder with the same name as the log. If `--summarize-scv` is on, then per-minibatch stats are written into a separate CSV file.
* At the end of evaluation on the testing set, the stats are written into a CSV file whose name ends in `Best.metrics.csv` (for the model checkpoint with the best validation score) and `Final.metrics.csv` (for the model checkpoint from the last epoch). If there are multiple runs whose prefixes only differ by text after a dash (e.g. `run-1, run-2, run-3`, etc.) then their metrics will be appended to the same CSV.
* Model outputs (predictions) are saved as .pt files in `predict/`. ROC curves are also saved there as CSV files. In the multiclass case, each class takes three columns of the file (the columns are [fpr, tpr, threshold]).
* Model checkpoints are saved as .pt files in `model/`.

### Executing the training scripts

* Here is an example of a command that starts training the classifier on the sample dataset that is part of this repository. Optimally there should be three files with names train.h5, valid.h5, and test.h5.
```
python3 train_pelican_classifier.py --datadir=./data/sample_data/run12 --yaml=./config/48k.yaml --target=is_signal --batch-size=64 --prefix=classifier
```
* Similarly for regression (this prompt will try to predict the top quark momentum stored in the sample dataset, however since 50% of the dataset are background events without a top quark, for which truth_Pmu=0, this is not a real regression test and will produce subpar predictions. You will also see NaNs in some of the metrics, which is expected.):
```
python3 train_pelican_cov.py --datadir=./data/sample_data/run12 --target=truth_Pmu --yaml=./config/48k.yaml --nobj=80 --nobj-avg=20 --batch-size=64 --prefix=regressor
```

For training on some custom datasets, such as Quark-Gluon and JetClass, there is an extra argument called `--dataset' which can be set to `qg` or `jc` to enable automatic reading in of the PID/charge information from those datasets.


## Authors

Alexander Bogatskiy, Flatiron Institute

Jan T. Offermann, University of Chicago 

Alexander Bogatskiy, Flatiron Institute
Jan T. Offermann, University of Chicago 
Timothy Hoffman, University of Chicago

Xiaoyang Liu, University of Chicago

## Acknowledgments

Inspiration, code snippets, etc.
* [Masked BatchNorm](https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py)
* [Gradual Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py)
* [whichcraft](https://github.com/cookiecutter/whichcraft)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details