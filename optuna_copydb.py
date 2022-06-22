import os
import optuna

if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-name', default='nosave')
    parser.add_argument('--host', default='worker1172')
    parser.add_argument('--password', default='asoetuh')
    parser.add_argument('--port', default='35719')
    parser.add_argument('--direction', default='download')

    args = parser.parse_args()
    
    storage_remote=f'postgresql://{os.environ["USER"]}:{args.password}@{args.host}:{args.port}'   # For running on nodes with a distributed file system
    storage_local='sqlite:///'+args.study_name+'.db'  # For running on a local machine

    if args.direction == 'download':
        study = optuna.copy_study(from_study_name=args.study_name, from_storage=storage_remote, to_storage=storage_local, to_study_name=args.study_name)
        print(f"Study {args.study_name} successfully copied to {storage_local}: ")
    elif args.direction == 'upload':
        study = optuna.copy_study(from_study_name=args.study_name, from_storage=storage_local, to_storage=storage_remote, to_study_name=args.study_name)
        print(f"Study {args.study_name} successfully copied to {storage_remote}: ")
