import logging
import os
import sys
import numpy
import random

from src.trainer import which
if which('nvidia-smi') is not None:
    min=40000
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

from src.models import PELICANRegression
from src.models import tests, expand_data, ir_data, irc_data
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from src.trainer import init_optimizer, init_scheduler
from src.models.metrics_cov import metrics, minibatch_metrics, minibatch_metrics_string, Angle3D
from src.models.metrics_cov import loss_fn_dR, loss_fn_pT, loss_fn_m, loss_fn_psi, loss_fn_inv, loss_fn_col, loss_fn_m2, loss_fn_3d, loss_fn_4d, loss_fn_E, loss_fn_col3
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
    model = PELICANRegression(args.num_channels_m, args.num_channels1, args.num_channels2, args.num_channels_m_out, num_targets=args.num_targets,
                      activate_agg=args.activate_agg, activate_lin=args.activate_lin,
                      activation=args.activation, add_beams=args.add_beams, config1=args.config1, config2=args.config2, average_nobj=args.nobj_avg,
                      factorize=args.factorize, masked=args.masked,
                      activate_agg2=args.activate_agg2, activate_lin2=args.activate_lin2, mlp_out=args.mlp_out,
                      scale=args.scale, ir_safe=args.ir_safe, c_safe=args.c_safe, dropout = args.dropout, drop_rate=args.drop_rate, drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
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
    loss_fn = lambda predict, targets:      0.05 * loss_fn_m(predict,targets) + 0.01 * loss_fn_3d(predict, targets) #0.01 * loss_fn_E(predict, targets) + 10 * loss_fn_psi(predict,targets) #  #    #+ 0.02 * loss_fn_E(predict,targets) #  #+  + 0.01 * loss_fn_pT(predict,targets) #  #0.03 * loss_fn_inv(predict,targets) + 
    # loss_fn = lambda predict, targets:      0.0005 * loss_fn_col(predict,targets) + 0.01*(-predict[...,0]).relu().mean() + 0.001 * loss_fn_inv(predict,targets) # 0.1 * loss_fn_m(predict,targets)

    # Apply the covariance and permutation invariance tests.
    if args.test:
        tests(model, dataloaders['train'], args, tests=['gpu','irc', 'permutation'], cov=True)

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
    if args.test:
        args.predict = False
        trainer.summarize_csv = False
        logger.info(f'EVALUATING BEST MODEL ON IR-SPLIT DATA (ADDED ONE 0-MOMENTUM PARTICLE)')
        trainer.evaluate(splits=['test'], final=False, ir_data=ir_data, expand_data=expand_data)
        logger.info(f'EVALUATING BEST MODEL ON IRC-SPLIT DATA (ADD A NEW PARTICLE SLOT AND SPLIT ONE BEAM INTO TWO EQUAL HALVES)')
        trainer.evaluate(splits=['test'], final=False, c_data=irc_data, expand_data=expand_data)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    main()
