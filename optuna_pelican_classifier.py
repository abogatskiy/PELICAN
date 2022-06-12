import torch
from torch.utils.data import DataLoader

import logging
import optuna
from optuna.trial import TrialState

from src.models import PELICANClassifier
from src.models import tests
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from src.trainer import init_optimizer, init_scheduler
from src.models.metrics_classifier import metrics, minibatch_metrics, minibatch_metrics_string

from src.dataloaders import initialize_datasets, collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')


def suggest_params(args, trial):

    args.lr_init = trial.suggest_loguniform("lr_init", 0.0001, 0.1)
    n_layers1 = trial.suggest_int("n_layers1", 1, 6)
    n_channels1 = trial.suggest_int("n_channels1", 5, 30)
    args.num_channels1 = [n_channels1,] * (n_layers1 + 1)
    n_layers_m = trial.suggest_int("n_layers_m", 0, 3)
    args.num_channels_m = [trial.suggest_int("n_channels_m["+str(i)+"]", 5, 30) for i in range(n_layers_m)]
    n_layers2 = trial.suggest_int("n_layers2", 1, 4)
    n_channels2 = trial.suggest_int("n_channels2", 5, 30)
    args.num_channels2 = [n_channels2,] * (n_layers2 + 1)
    args.activation = trial.suggest_categorical("activation", ["relu", "elu", "leakyrelu", "silu", "selu", "tanh"])
    args.optim = trial.suggest_categorical("optim", ["adamw", "sgd", "amsgrad", "rmsprop", "adam"])
    args.config = trial.suggest_categorical("config", ["s", "S", "m", "M", "sS", "mM", "sm", "sM", "Sm", "SM", "sSm", "sSM", "smM", "sMmM", "mx", "Mx", "mxn", "mXN", "mxMX", "sXN", "smxn"])

    return args

def define_model(trial):
   
    # Initialize arguments
    args = init_argparse()

    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Write input paramaters and paths to log
    logging_printout(args)
        # Suggest parameters to optuna to optimize over
    args = suggest_params(args, trial)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize model
    model = PELICANClassifier(args.num_channels0, args.num_channels_m, args.num_channels1, args.num_channels2,
                      message_depth=args.message_depth, activation=args.activation, add_beams=args.add_beams, sym=args.sym, config=args.config,
                      scale=1., ir_safe=args.ir_safe, dropout = args.dropout, batchnorm=args.batchnorm,
                      device=device, dtype=dtype)

    model.to(device)

    return args, model, device, dtype

def define_dataloader(args):

    # Initialize dataloder
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None)

    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj, add_beams=args.add_beams, beam_mass=args.beam_mass)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate)
                   for split, dataset in datasets.items()}

    return args, dataloaders


def objective(trial):

    args, model, device, dtype = define_model(trial)

    args, dataloaders = define_dataloader(args)

    if args.parallel:
        model = torch.nn.DataParallel(model)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs, summarize = init_scheduler(args, optimizer)

    # Define a loss function.
    # loss_fn = torch.nn.functional.cross_entropy
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    
    # Apply the covariance and permutation invariance tests.
    if args.test:
        tests(model, dataloaders['train'], args, tests=['permutation','batch','irc'])

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, metrics, minibatch_metrics, minibatch_metrics_string, optimizer, scheduler, restart_epochs, summarize, device, dtype)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.
    trainer.train(trial=trial)

    best_loss = torch.load(args.bestfile)['best_loss']

    # # Test predictions on best model and also last checkpointed model.
    # best_loss = trainer.evaluate(splits=['test'])

    return best_loss

if __name__ == '__main__':

    # Initialize arguments
    args = init_argparse()

    study = optuna.create_study(study_name=args.study_name, storage='sqlite:///'+args.study_name+'.db', direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
