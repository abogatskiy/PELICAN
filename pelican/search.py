"""
Optuna hyperparameter search (pelican optuna ...). Each trial is a normal training run in
<runs-dir>/<study-name>-trial<k>/. With --storage remote, trials coordinate through the postgres server started
by slurm/run_db.sh (see slurm/OptunaRustyReadme.md). Edit `suggest` to change the search space.
"""
import copy
import logging
import os

logger = logging.getLogger('pelican')


def suggest(trial, cfg):
    cfg.lr_init = trial.suggest_float('lr_init', 1e-4, 1e-1, log=True)
    cfg.drop_rate = trial.suggest_float('drop_rate', 0, 0.5, step=0.005)
    cfg.weight_decay = trial.suggest_float('weight_decay', 0, 0.01, step=0.001)
    cfg.batch_size = trial.suggest_categorical('batch_size', [32, 128, 512])
    cfg.factorize = trial.suggest_categorical('factorize', [False, True])
    n_out = trial.suggest_int('n_layers_m_out', 1, 2)
    cfg.num_channels_m_out = [trial.suggest_int(f'n_channels_m_out[{k}]', 10, 60) for k in range(n_out)]
    n_blocks = trial.suggest_int('n_blocks', 2, 8)
    cfg.num_channels_eq = [trial.suggest_int(f'n_channels_eq[{i}]', 5, 60) for i in range(n_blocks)]
    width_m = trial.suggest_int('n_channels_m', max(10, cfg.num_channels_scalar + 1), 60)    # must exceed the scalar channels
    cfg.num_channels_m = [[width_m] for _ in range(n_blocks)]
    cfg.drop_rate_out = trial.suggest_float('drop_rate_out', 0, 0.25, step=0.05)
    n_layers_out = trial.suggest_int('n_layers_out', 1, 2)
    cfg.num_channels_out = [trial.suggest_int(f'n_channels_out[{i}]', 10, 60) for i in range(n_layers_out)]
    cfg.agg = trial.suggest_categorical('agg', ['s', 'm', 'S', 'M'])
    return cfg


def _storage(cfg):
    import optuna
    if cfg.storage == 'remote':
        return optuna.storages.RDBStorage(url=f'postgresql://{os.environ["USER"]}:{cfg.password}@{cfg.host}:{cfg.port}',
                                          heartbeat_interval=100)
    if cfg.storage == 'local':
        return f'sqlite:///file:{cfg.study_name}.db?vfs=unix-dotfile&uri=true'
    return cfg.storage


def run_study(cfg):
    import optuna
    from optuna.trial import TrialState
    from .cli import cmd_train

    minimize = cfg.metric == 'loss'
    sampler = optuna.samplers.RandomSampler() if cfg.sampler == 'random' else \
        optuna.samplers.TPESampler(n_startup_trials=100, multivariate=True, group=True, constant_liar=True)
    pruner = {'median': optuna.pruners.MedianPruner(n_warmup_steps=20, n_min_trials=10),
              'hyperband': optuna.pruners.HyperbandPruner(), 'none': optuna.pruners.NopPruner()}[cfg.pruner]
    study = optuna.create_study(study_name=cfg.study_name, storage=_storage(cfg), load_if_exists=True,
                                direction='minimize' if minimize else 'maximize', sampler=sampler, pruner=pruner)

    def objective(trial):
        tcfg = suggest(trial, copy.deepcopy(cfg))
        tcfg.prefix = f'{cfg.study_name}-trial{trial.number}'
        tcfg.run_dir = os.path.join(tcfg.runs_dir, tcfg.prefix)
        tcfg.command = 'train'
        best_epoch, best_metrics, results = cmd_train(tcfg, trial=trial)
        trial.set_user_attr('best_epoch', best_epoch)
        trial.set_user_attr('best_metrics', best_metrics)
        if results:
            trial.set_user_attr('test_metrics', {f'{w}_{s}': m for (w, s), m in results.items()})
        return best_metrics[cfg.metric]

    study.optimize(objective, n_trials=cfg.n_trials,
                   callbacks=[optuna.study.MaxTrialsCallback(cfg.n_trials, states=(TrialState.COMPLETE,))])
    complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    logger.info(f'{len(study.trials)} trials ({len(complete)} complete, {len(pruned)} pruned)')
    if complete:
        best = study.best_trial
        logger.info(f'best trial {best.number}: {cfg.metric} = {best.value}')
        for k, v in best.params.items():
            logger.info(f'  {k}: {v}')
