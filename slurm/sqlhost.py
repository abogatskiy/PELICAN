import os
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--password', default='asoetuh')
    parser.add_argument('--port', default='5432')

    ns = parser.parse_args()

    study = optuna.create_study(study_name='my-study', storage=f'postgresql://{os.environ["USER"]}:{ns.password}@{ns.host}:{ns.port}', load_if_exists=True)
    study.optimize(objective, n_trials=20)

if __name__ == '__main__':
    main()