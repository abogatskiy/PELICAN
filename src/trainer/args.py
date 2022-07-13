import argparse

from math import inf

#### Argument parser ####

def setup_argparse():

    parser = argparse.ArgumentParser(description='LGN network options')

    parser.add_argument('--host', default='worker1172')
    parser.add_argument('--password', default='asoetuh')
    parser.add_argument('--port', default='35719')
    parser.add_argument('--storage', default='local', help="specify location of storage to use for Optuna (\'remote\' | \'local\' | specify path) (default: local)")
    parser.add_argument('--sampler', default='random')
    parser.add_argument('--pruner', default='median')

    parser.add_argument('--task', type=str, default='train', metavar='str',
                        help='Train or evaluate model. (train | eval)')

    # Optimizer options
    parser.add_argument('--num-epoch', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', '-bs', type=int, default=10, metavar='N',
                        help='Mini-batch size (default: 10)')
    parser.add_argument('--batch-group-size', '-bgs', type=int, default=1, metavar='N',
                        help='Mini-batch size (default: 10)')    

    parser.add_argument('--weight-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer (default: 0)')
    parser.add_argument('--lr-init', type=float, default=0.0015, metavar='N',
                        help='Initial learning rate (default: 0.005)')
    parser.add_argument('--lr-final', type=float, default=5e-6, metavar='N',
                        help='Final (held) learning rate (default: 1e-5)')
    parser.add_argument('--lr-decay', type=int, default=-1, metavar='N',
                        help='Timescale over which to decay the learning rate (-1 to disable) (default: -1)')
    parser.add_argument('--lr-decay-type', type=str, default='cos', metavar='str',
                        help='Type of learning rate decay. (cos | linear | exponential | pow | restart) (default: cos)')
    parser.add_argument('--lr-minibatch', '--lr-mb', action=argparse.BooleanOptionalAction, default=True,
                        help='Decay learning rate every minibatch instead of epoch.')
    parser.add_argument('--sgd-restart', type=int, default=-1, metavar='int',
                        help='Restart SGD optimizer every (lr_decay)^p epochs, where p=sgd_restart. (-1 to disable) (default: -1)')
    parser.add_argument('--optuna-test', action=argparse.BooleanOptionalAction, default=False,
                        help='Run best epoch\'s model on testing set at the end of an Optuna trial.')

    parser.add_argument('--optim', type=str, default='adamw', metavar='str',
                        help='Set optimizer. (SGD, AMSgrad, Adam, AdamW, RMSprop)')
    parser.add_argument('--parallel', action=argparse.BooleanOptionalAction, default=False,
                        help='Use nn.DataParallel when multiple GPUs are available.')
    parser.add_argument('--summarize', action=argparse.BooleanOptionalAction, default=True,
                        help='Use a TensorBoard SummaryWriter() to log metrics.')

    # Dataloader and randomness options
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True,
                        help='Shuffle minibatches.')
    parser.add_argument('--seed', type=int, default=-1, metavar='N',
                        help='Set random number seed. Set to -1 to set based upon clock.')

    # Saving and logging options
    parser.add_argument('--alpha', type=float, default=0, metavar='N',
                    help='Averaging exponent for recent loss printouts [0, inf), the higher the more smoothing (default = 0')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True,
                        help='Save checkpoint after each epoch. (default: True)')
    parser.add_argument('--load', action=argparse.BooleanOptionalAction, default=False,
                        help='Load from previous checkpoint. (default: False)')

    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False,
                        help='Perform automated network testing. (Default: False)')

    parser.add_argument('--log-level', type=str, default='info',
                        help='Logging level to output')

    parser.add_argument('--textlog', action=argparse.BooleanOptionalAction, default=True,
                        help='Log a summary of each mini-batch to a text file.')

    parser.add_argument('--predict', action=argparse.BooleanOptionalAction, default=True,
                        help='Save predictions. (default: True)')

    parser.add_argument('--quiet', action=argparse.BooleanOptionalAction, default=True,
                        help='Hide warnings about unused parameters. (default: True)')

    ### Arguments for files to save things to
    # Job prefix is used to name checkpoint/best file
    parser.add_argument('--prefix', '--jobname', type=str, default='nosave',
                        help='Prefix to set load, save, and logfile. (default: nosave)')
    parser.add_argument('--study-name', type=str, default='nosave',
                        help='Name for the optuna study. (default: nosave)')

    # Allow to manually specify file to load
    parser.add_argument('--loadfile', type=str, default='',
                        help='Set checkpoint file to load. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save model checkpoint to
    parser.add_argument('--checkfile', type=str, default='',
                        help='Set checkpoint file to save checkpoints to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to best model checkpoint to
    parser.add_argument('--bestfile', type=str, default='',
                        help='Set checkpoint file to best model to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save logging information to
    parser.add_argument('--logfile', type=str, default='',
                        help='Duplicate logging.info output to logfile. Set to empty string to generate from prefix. (default: (empty))')
    # Verbose logging toggle to reduce log file size (still prints minibatches on screen)
    parser.add_argument('--verbose', '-v', action=argparse.BooleanOptionalAction, default=False,
                        help='Write per-minibatch info to logfile. (default: True)')
    # Filename to save predictions to
    parser.add_argument('--predictfile', type=str, default='',
                        help='Save predictions to file. Set to empty string to generate from prefix. (default: (empty))')

    # Working directory to place all files
    parser.add_argument('--workdir', type=str, default='./',
                        help='Working directory as a default location for all files. (default: ./)')
    # Directory to place logging information
    parser.add_argument('--logdir', type=str, default='log/',
                        help='Directory to place log and savefiles. (default: log/)')
    # Directory to place saved models
    parser.add_argument('--modeldir', type=str, default='model/',
                        help='Directory to place log and savefiles. (default: model/)')
    # Directory to place model predictions
    parser.add_argument('--predictdir', type=str, default='predict/',
                        help='Directory to place log and savefiles. (default: predict/)')
    # Directory to read and save data from
    parser.add_argument('--datadir', type=str, default='data/',
                        help='Directory to look up data from. (default: data/)')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='jet',
                        help='Data set. Options: (jet, 2v3bodycomplex). Default: jet.')
    parser.add_argument('--target', type=str, default='is_signal',
                        help='Learning target for a dataset (such as qm9) with multiple options.')
    
    parser.add_argument('--nobj', type=int, default=None, metavar='N',
                        help='Max number of particles in each event (selects the first nobj). Set to None to use entire dataset. (default: None)')
    parser.add_argument('--num-train', type=int, default=-1, metavar='N',
                        help='Number of samples to train on. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-valid', type=int, default=-1, metavar='N',
                        help='Number of validation samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-test', type=int, default=-1, metavar='N',
                        help='Number of test samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--add-beams', action=argparse.BooleanOptionalAction, default=True,
                        help='Append two proton beams of the form (m^2,0,0,+-1) to each event')
    parser.add_argument('--beam-mass', type=float, default=1, metavar='N',
                    help='Set mass m of the beams, so that E=sqrt(1 + m^2) (default = 1)')
    parser.add_argument('--force-download', action=argparse.BooleanOptionalAction, default=False,
                        help='Force download and processing of dataset.')

    # Computation options
    parser.add_argument('--device', type=str, default='cuda',
                    help='Which device to use (cpu | gpu/cuda | mps/m1) (default = cuda)')
    parser.add_argument('--cuda', '--gpu', dest='device', action='store_const', const='cuda',
                        help='Use CUDA (default)')
    parser.add_argument('--cpu', dest='device', action='store_const', const='cpu',
                        help='Use CPU')
    parser.add_argument('--mps', '--m1', dest='device', action='store_const', const='mps',
                        help='Use M1 chip [Experimental]')

    parser.add_argument('--float', dest='dtype', action='store_const', const='float',
                        help='Use floats.')
    parser.add_argument('--double', dest='dtype', action='store_const', const='double',
                        help='Use doubles.')
    parser.set_defaults(dtype='float')

    parser.add_argument('--num-workers', type=int, default=0,
                        help='Set number of workers in dataloader. (Default: 0)')

    # Model options

    parser.add_argument('--num-channels0', nargs='*', type=int, default=[5,]*2, metavar='N',
                        help='Number of channels to allow after mixing (default: [3])')
    parser.add_argument('--num-channels-m', nargs='*', type=int, metavar='N',
                        help='Number of channels to allow after mixing (default: [3])',
                        # default=[[38], [24], [11], [24], [32], [40]]
                        default =[[40],[25],[10],[25],[30],[40]]
                        )
    parser.add_argument('--num-channels1', nargs='*', type=int, metavar='N',
                        help='Number of channels to allow after mixing (dfault: [3])',
                        # default=[37,20,19,16,21,22,23]
                        default=[35,20,20,15,20,20,35]
                        )
    parser.add_argument('--num-channels2', nargs='*', type=int, default=[30], metavar='N',
                        help='Number of channels to allow after mixing (default: [3])')
    parser.add_argument('--dropout', action=argparse.BooleanOptionalAction, default=True,
                    help='Enable a dropout layer at the end of the network (default = False)')
    parser.add_argument('--drop-rate', type=float, default=0.25, metavar='N',
                    help='Dropout rate (default = 0.2)')
    parser.add_argument('--batchnorm', type=str, default='b',
                    help='Enable batch/instance normalization at the end of each MessageNet (batch | instance | None) (default = b)')
    parser.add_argument('--layernorm', action=argparse.BooleanOptionalAction, default=False,
                    help='Enable layer normalization in the input layer (default = False)')
    parser.add_argument('--sig', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable message significance networks (default = False)')
    parser.add_argument('--config1', type=str, default='s',
                    help='Configuration for aggregation functions in Net2to2 (any combination of letters s,S,m,M,x,X,n,N (default = s)')
    parser.add_argument('--config2', type=str, default='s',
                    help='Configuration for aggregation functions in Eq2to0 (any combination of letters s,S,m,M,x,X,n,N (default = s)')
    parser.add_argument('--activate-agg', action=argparse.BooleanOptionalAction, default=False,
                    help='Apply an activation function right after permutation-equvariant aggregation (default = False)')
    parser.add_argument('--activate-lin', action=argparse.BooleanOptionalAction, default=True,
                    help='Apply an activation function right after the linear mixing following aggregation (default = True)')


    parser.add_argument('--scale', type=float, default=0.33, metavar='N',
                    help='Global scaling factor for input four-momenta (default = 1.)')

    parser.add_argument('--activation', type=str, default='leakyrelu',
                        help='Activation function used in MLP layers. Options: (relu, elu, leakyrelu, sigmoid, logsigmoid, atan, silu, celu, selu, soft, tanh). Default: elu.')                
    parser.add_argument('--ir-safe', action=argparse.BooleanOptionalAction, default=False,
                    help='Use an IRC safe version of the model (injecting extra particles with small momenta won\'t change the outputs) (default = False)')

    # TODO: Update(?)
    parser.add_argument('--weight-init', type=str, default='randn', metavar='str',
                        help='Weight initialization function to use (default: rand)')

    return parser


# From https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
