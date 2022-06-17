import torch
from torch.utils.data import DataLoader

import os
import optuna


# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)


if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-name', default='nosave')
    parser.add_argument('--host', default='worker1172')
    parser.add_argument('--password', default='asoetuh')
    parser.add_argument('--port', default='35719')

    args = parser.parse_args()
    
    from_storage=f'postgresql://{os.environ["USER"]}:{args.password}@{args.host}:{args.port}'   # For running on nodes with a distributed file system
    to_storage='sqlite:///file:'+args.study_name+'.db'  # For running on a local machine

    study = optuna.copy_study(from_study_name=args.study_name, from_storage=from_storage, to_storage=to_storage, to_study_name=args.study_name)
  
    print(f"Study {args.study_name} successfully copied to {to_storage}: ")
