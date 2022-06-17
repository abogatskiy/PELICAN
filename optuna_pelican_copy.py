import torch
from torch.utils.data import DataLoader

import os
import optuna


# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)


if __name__ == '__main__':

    args = init_argparse()
    
    from_storage=f'postgresql://{os.environ["USER"]}:{args.password}@{args.host}:{args.port}'   # For running on nodes with a distributed file system
    to_storage='sqlite:///file:'+args.study_name+'.db?vfs=unix-dotfile&uri=true'  # For running on a local machine

    study = optuna.copy_study(from_study_name=args.study_name, from_storage=storage, to_storage=to_storage, to_study_name=args.study_name)
  
    print(f"Study {args.study_name} successfully copied to {to_storage}: ")
