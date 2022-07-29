import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as sched

import os, sys, pickle
from datetime import datetime
from math import inf, log, log2, exp, ceil


import logging
from colorlog import ColoredFormatter

logger = logging.getLogger(__name__)

# if sys.version_info < (3, 6):
#     logger.info('LGN requires Python version 3.6! or above!')
#     sys.exit(1)

MAE = torch.nn.L1Loss()
MSE = torch.nn.MSELoss()
RMSE = lambda x, y : sqrt(MSE(x, y))

#### Initialize parameters for training run ####

def init_argparse():
    from .args import setup_argparse

    parser = setup_argparse()
    args = parser.parse_args()

    return args

def init_logger(args):
    if args.logfile:
        handlers = [logging.FileHandler(args.logfile, mode='a' if args.load else 'w'), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    if args.log_level.lower() == 'debug':
        loglevel = logging.DEBUG
    elif args.log_level.lower() == 'info':
        loglevel = logging.INFO
    else:
        ValueError('Inappropriate choice of logging_level. {}'.format(args.logging_level))

    logging.basicConfig(level=loglevel,
                        format="%(message)s",
                        handlers=handlers
                        )

    LOGFORMAT = "  %(log_color)s%(message)s%(reset)s"
    logging.root.setLevel(loglevel)
    formatter = ColoredFormatter(LOGFORMAT)
    handlers[-1].setLevel(loglevel)
    handlers[-1].setFormatter(formatter)
    # logger.setLevel(loglevel)
    # logger.addHandler(handlers[-1])

def fix_args(args):
    if args.task.startswith('eval'):
        args.load = True
        args.num_epoch = 0
    if len(args.num_channels_m)==0:
        args.num_channels_m = [[] for i in range(len(args.num_channels1))]
        print("MessageNet was disabled for all equivariant layers")
    elif type(args.num_channels_m[0])==int:
        args.num_channels_m = [args.num_channels_m,] * len(args.num_channels1)
        print("MessageNet hyperparams are the same across all equivariant layers")
        # args.num_channels_m[0][0] = args.num_channels1[0] # delete this line if not using Residual connections
    
    if args.seed < 0:
        seed = int((datetime.now().timestamp())*100000)
        logger.info('Setting seed based upon time: {}'.format(seed))
        args.seed = seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.reproducible:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    
    return args

def init_file_paths(args):

    # Initialize files and directories to load/save logs, models, and predictions
    workdir = args.workdir
    prefix = args.prefix
    modeldir = args.modeldir
    logdir = args.logdir
    predictdir = args.predictdir

    if not args.loadfile:
        if prefix and not args.logfile:  args.logfile =  os.path.join(workdir, logdir, prefix+'.log')
        if prefix and not args.bestfile: args.bestfile = os.path.join(workdir, modeldir, prefix+'_best.pt')
        if prefix and not args.checkfile: args.checkfile = os.path.join(workdir, modeldir, prefix+'.pt')
        if prefix and not args.predictfile: args.predictfile = os.path.join(workdir, predictdir, prefix)
        if prefix: args.loadfile = args.checkfile
    elif os.path.exists(args.loadfile):
        if prefix and not args.logfile:  args.logfile =  os.path.join(workdir, logdir, prefix+'.log')
        if prefix and not args.bestfile: args.bestfile = args.loadfile
        if prefix and not args.checkfile: args.checkfile = args.loadfile
        if prefix and not args.predictfile: args.predictfile = os.path.join(workdir, predictdir, prefix)
        args.load = True
    else:
        logger.info('The specified loadfile {} doesn\'t exist!'.format(args.loadfile))

    if not os.path.exists(os.path.join(workdir, modeldir)):
        logger.warning('Model directory {} does not exist. Creating!'.format(modeldir))
        os.mkdir(modeldir)
    if not os.path.exists(os.path.join(workdir, logdir)):
        logger.warning('Logging directory {} does not exist. Creating!'.format(logdir))
        os.mkdir(logdir)
    if not os.path.exists(os.path.join(workdir, predictdir)):
        logger.warning('Prediction directory {} does not exist. Creating!'.format(predictdir))
        os.mkdir(predictdir)

    args.dataset = args.dataset.lower()

    return args

def logging_printout(args, trial=None):
    
    # Printouts of various inputs before training (and after logger is initialized with correct logfile path)
    logger.info('Initializing simulation based upon argument string:')
    logger.info(' '.join([arg for arg in sys.argv]))
    logger.info('Log, best, checkpoint, load files: {} {} {} {}'.format(args.logfile, args.bestfile, args.checkfile, args.loadfile))
    logger.info('Dataset, learning target, datadir: {} {} {}'.format(args.dataset, args.target, args.datadir))
    git_status = _git_version()


    logger.info('Values of all model arguments:')
    logger.info('{}'.format(args))
    if trial:    
        logger.info('Params of Optuna trial:')
        for key, val in trial.params.items():
            logger.info(f'\'{key}\': {val}')

    return git_status


#### Initialize optimizer ####

def init_optimizer(args, model):

    params = {'params': model.parameters(), 'lr': args.lr_init, 'weight_decay': args.weight_decay}
    params = [params]

    optim_type = args.optim.lower()

    if optim_type == 'adam':
        optimizer = optim.Adam(params, amsgrad=False)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(params, amsgrad=False)
    elif optim_type == 'amsgrad':
        optimizer = optim.Adam(params, amsgrad=True)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(params)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(params)
    else:
        raise ValueError('Incorrect choice of optimizer')

    return optimizer

def init_scheduler(args, optimizer):
    lr_init, lr_final = args.lr_init, args.lr_final
    if args.lr_decay < 0: 
        lr_decay = args.num_epoch
    else:
        lr_decay = min(args.lr_decay, args.num_epoch)
        
    minibatch_per_epoch = ceil(args.num_train / args.batch_size)
    if args.lr_minibatch:
        lr_decay = lr_decay*minibatch_per_epoch

    lr_ratio = lr_final/lr_init

    lr_bounds = lambda lr, lr_min: min(1, max(lr_min, lr))

    if args.sgd_restart > 0:
        restart_epochs = [(2**k-1) for k in range(1, ceil(log2(args.num_epoch))+1)]
        lr_hold = restart_epochs[0]
        if args.lr_minibatch:
            lr_hold *= minibatch_per_epoch
        logger.info('SGD Restart epochs: {}'.format(restart_epochs))
    else:
        restart_epochs = []
        lr_hold = args.num_epoch
        if args.lr_minibatch:
            lr_hold *= minibatch_per_epoch

    if args.lr_decay_type.startswith('cos'):
        scheduler = sched.CosineAnnealingLR(optimizer, lr_hold, eta_min=lr_final)
    elif args.lr_decay_type.startswith('warm'):
        scheduler = sched.CosineAnnealingWarmRestarts(optimizer, 4*minibatch_per_epoch, 2)
    elif args.lr_decay_type.startswith('exp'):
        # lr_lambda = lambda epoch: lr_bounds(exp(epoch / lr_decay * log(lr_ratio)), lr_ratio)
        # scheduler = sched.LambdaLR(optimizer, lr_lambda)
        scheduler = sched.ExponentialLR(optimizer, exp(log(lr_ratio) / lr_decay))
    elif args.lr_decay_type.startswith('one'):
        scheduler = sched.OneCycleLR(optimizer, 10 * lr_init, epochs=args.num_epoch, steps_per_epoch=minibatch_per_epoch)
    else:
        raise ValueError('Incorrect choice for lr_decay_type!')

    return scheduler, restart_epochs, args.summarize_csv, args.summarize

#### Other initialization ####

def _git_version():
    from subprocess import run, PIPE
    git_commit = run('git log --pretty=%h -n 1'.split(), stdout=PIPE)
    logger.info('Git status: {}'.format(git_commit.stdout.decode()))
    logger.info('Date and time: {}'.format(datetime.now()))

    return str(git_commit.stdout.decode())

def init_cuda(args):
    if args.device in ['gpu', 'cuda']:
        assert(torch.cuda.is_available()), "No CUDA device available!"
        logger.info('Initializing CUDA/GPU! Device: {}'.format(torch.cuda.current_device()))
        torch.cuda.init()
        device = torch.device('cuda')
    elif args.device in ['mps', 'm1']:
        logger.info('Initializing M1 Chip!')
        device = torch.device('mps')
    else:
        logger.info('Initializing CPU!')
        device = torch.device('cpu')

    if args.dtype == 'double':
        dtype = torch.double
    elif args.dtype == 'float':
        dtype = torch.float
    else:
        raise ValueError('Incorrect data type chosen!')

    return device, dtype
