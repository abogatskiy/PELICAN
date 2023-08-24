import logging
import os
import sys
import numpy
import random

from src.trainer import which
if which('nvidia-smi') is not None:
    min=8000
    deviceid = 0
    name, mem = os.popen('"nvidia-smi" --query-gpu=gpu_name,memory.total --format=csv,nounits,noheader').read().split('\n')[deviceid].split(',')
    print(mem)
    mem = int(mem)
    if mem < min:
        print('Less GPU memory than requested. Terminating.')
        sys.exit()

logger = logging.getLogger('')

import torch
from torch.utils.data import DataLoader

from src.models import PELICANMass
from src.models import tests
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from src.trainer import init_optimizer, init_scheduler
from src.models.metrics_mass import metrics, minibatch_metrics, minibatch_metrics_string
from src.models.lorentz_metric import normsq4, dot4

from src.dataloaders import initialize_datasets, collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000, sci_mode=False)


def main():

    # Initialize arguments -- Just
    args = init_argparse()
   
    # Initialize file paths
    args = init_file_paths(args)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Initialize logger
    init_logger(args)

    if which('nvidia-smi') is not None:
        logger.info(f'Using {name} with {mem} MB of GPU memory')
    
    # Write input paramaters and paths to log
    logging_printout(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloder
    if args.fix_data:
        torch.manual_seed(165937750084982)
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None, testfile=args.testfile)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj, add_beams=args.add_beams, beam_mass=args.beam_mass)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker,
                                     collate_fn=collate)
                   for split, dataset in datasets.items()}

    # Initialize model
    model = PELICANMass(args.num_channels_m, args.num_channels_2to2, args.num_channels_out, args.num_channels_m_out,
                      activate_agg=args.activate_agg, activate_lin=args.activate_lin,
                      activation=args.activation, add_beams=args.add_beams, config=args.config, config_out=args.config_out, average_nobj=args.nobj_avg,
                      factorize=args.factorize, masked=args.masked,
                      activate_agg_out=args.activate_agg_out, activate_lin_out=args.activate_lin_out, mlp_out=args.mlp_out,
                      scale=args.scale, ir_safe=args.ir_safe,
                        # c_safe=args.c_safe,
                      dropout = args.dropout, drop_rate=args.drop_rate, drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
                      device=device, dtype=dtype)
    
    model.to(device)

    if args.parallel:
        model = torch.nn.DataParallel(model)

    # Initialize the scheduler and optimizer
    if args.task.startswith('eval'):
        optimizer = scheduler = None
        restart_epochs = []
        args.summarize = False
    else:
        optimizer = init_optimizer(args, model, len(dataloaders['train']))
        scheduler, restart_epochs = init_scheduler(args, optimizer)


    # Define a loss function. This is the loss function whose gradients are actually computed. 

    loss_fn_m = lambda predict, targets: (predict - normsq4(targets).abs().sqrt()).abs().mean()
    loss_fn = lambda predict, targets: 0.1 * loss_fn_m(predict,targets)

    # Apply the covariance and permutation invariance tests.
    if args.test:
        raise NotImplementedError()

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, metrics, minibatch_metrics, minibatch_metrics_string, optimizer, scheduler, restart_epochs, args.summarize_csv, args.summarize, device, dtype)

    if not args.task.startswith('eval'):
        # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
        trainer.load_checkpoint()
        # Set a CUDA variale that makes the results exactly reproducible on a GPU (on CPU they're reproducible regardless)
        if args.reproducible:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # Train model.
        trainer.train()

    # Test predictions on best model and/or also last checkpointed model.
    trainer.evaluate(splits=['test'], final=False)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    main()
