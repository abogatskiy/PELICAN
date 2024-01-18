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
        print(f'Less GPU memory than requested ({mem}<{min}). Terminating.')
        sys.exit()


import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from src.models import PELICANRegression
from src.models import tests, expand_data, ir_data, irc_data
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args, set_seed, get_world_size
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

    # args = fix_args(args)

    if 'LOCAL_RANK' in os.environ:
        device_id = int(os.environ["LOCAL_RANK"])
    else:
        device_id = -1


    # Initialize logger
    logger = logging.getLogger('')
    init_logger(args, device_id)

    if which('nvidia-smi') is not None:
        logger.info(f'Using {name} with {mem} MB of GPU memory (local rank {device_id})')
    
    # Write input paramaters and paths to log
    if device_id <= 0:
        logging_printout(args)

    # Initialize device and data type
    device, dtype = init_cuda(args, device_id)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)
    # Set a manual random seed for torch, cuda, numpy, and random
    args = set_seed(args, device_id)

    distributed = (get_world_size() > 1)
    if distributed:
        world_size = dist.get_world_size()
        logger.info(f'World size {world_size}')

    # Initialize dataloder
    if args.fix_data:
        torch.manual_seed(165937750084982)
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None, testfile=args.testfile, RAMdataset=args.RAMdataset)

    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj, add_beams=args.add_beams, beam_mass=args.beam_mass)
    if distributed:
        samplers = {'train': DistributedSampler(datasets['train'], shuffle=args.shuffle),
                    'valid': DistributedSampler(datasets['valid'], shuffle=False),
                    'test': None}
    else:
        samplers = {split: None for split in datasets.keys()}

    dataloaders = {split: DataLoader(dataset,
                                     batch_size = args.batch_size,
                                     shuffle = args.shuffle if (split == 'train' and not distributed) else False,
                                     num_workers = args.num_workers,
                                     pin_memory=True,
                                     worker_init_fn = seed_worker,
                                     collate_fn =collate,
                                     sampler = samplers[split]
                                     )
                   for split, dataset in datasets.items()}

    # Initialize model
    model = PELICANRegression(args.num_channels_scalar, args.num_channels_m, args.num_channels_2to2, args.num_channels_out, args.num_channels_m_out, num_targets=args.num_targets,
                      activate_agg=args.activate_agg, activate_lin=args.activate_lin,
                      activation=args.activation, add_beams=args.add_beams, read_pid=args.read_pid, config=args.config, config_out=args.config_out, average_nobj=args.nobj_avg,
                      factorize=args.factorize, masked=args.masked,
                      activate_agg_out=args.activate_agg_out, activate_lin_out=args.activate_lin_out, mlp_out=args.mlp_out,
                      scale=args.scale, irc_safe=args.irc_safe, dropout = args.dropout, drop_rate=args.drop_rate, drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
                      device=device, dtype=dtype)
    
    model.to(device)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[device_id])

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
    if args.test and device_id <= 0:
        with torch.autograd.set_detect_anomaly(True):
            tests(model, dataloaders['train'], args, tests=['gpu','irc', 'permutation'], cov=True)

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, metrics,
                      minibatch_metrics, minibatch_metrics_string, optimizer, scheduler,
                      restart_epochs, args.summarize_csv, args.summarize, device_id, device, dtype,
                      warmup_epochs=args.warmup, cooldown_epochs=args.cooldown)

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
    if distributed:
        dist.destroy_process_group()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    main()
