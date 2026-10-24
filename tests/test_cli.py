"""End-to-end runs of the command line on tiny synthetic HDF5 files."""
import importlib.util
import json
import os

import h5py
import numpy as np
import pytest
import torch

from pelican.cli import main, build_model
from conftest import make_jets

SMALL = ['--num-channels-m', '12', '--num-channels-eq', '8', '8', '--num-channels-m-out', '12', '8', '--num-channels-out', '8',
         '--num-channels-scalar', '4', '--nobj-avg', '10']
HERE = os.path.dirname(__file__)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, '..', 'scripts', f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def datadir(tmp_path_factory):
    d = tmp_path_factory.mktemp('data')
    for split, n, seed in [('train', 64, 0), ('valid', 32, 1), ('test', 32, 2)]:
        b = make_jets(B=n, N=30, seed=seed)
        with h5py.File(d / f'jets_{split}.h5', 'w') as f:
            f['Pmu'] = b['Pmu'].numpy()
            f['Nobj'] = b['Nobj'].numpy()
            f['is_signal'] = np.arange(n) % 2
            f['truth_Pmu'] = b['truth_Pmu'].numpy()
    return str(d)


def run(datadir, tmp_path, prefix, *extra):
    args = ['--datadir', datadir, '--runs-dir', str(tmp_path / 'runs'), '--prefix', prefix, '--cpu', '--num-epoch', '3',
            '--warmup', '1', '--cooldown', '1', '--batch-size', '16', '--seed', '1', *extra]
    return args


@pytest.mark.parametrize('task,extra', [
    ('classification', SMALL),
    ('regression', SMALL + ['--target', 'truth_Pmu']),
    ('classification', SMALL + ['--method', 'input', '--stabilizer', 'so2']),
    ('regression', SMALL + ['--method', 'input', '--stabilizer', '11', '--target', 'truth_Pmu']),
    ('classification', SMALL + ['--stabilizer', 'so13', '--clip-grad', '1']),
    ('classification', SMALL + ['--irc-safe', '--agg', 's', '--agg-out', 's']),
    ('classification', SMALL + ['--no-mlp-out', '--agg', 'x', '--agg-out', 'n']),
])
def test_train_then_eval(datadir, tmp_path, task, extra):
    prefix = 'run'
    main(['train', *run(datadir, tmp_path, prefix, '--task', task, '--self-test', *extra)])
    rd = tmp_path / 'runs' / prefix
    for f in ['config.yaml', 'train.log', 'checkpoint.pt', 'best.pt', 'metrics.csv', 'test_metrics.json']:
        assert (rd / f).exists(), f
    assert (rd / 'predictions' / 'best_test.pt').exists()
    metrics = json.loads((rd / 'test_metrics.json').read_text())
    assert np.isfinite(metrics['best_test']['loss'])
    lines = (rd / 'metrics.csv').read_text().strip().splitlines()
    assert len(lines) == 1 + 3                              # header + one row per epoch
    # re-evaluate from the run folder alone
    main(['eval', str(rd), '--cpu'])
    assert json.loads((rd / 'test_metrics.json').read_text())['best_test']['loss'] == pytest.approx(metrics['best_test']['loss'])


def test_refuses_to_overwrite_and_resumes(datadir, tmp_path):
    main(['train', *run(datadir, tmp_path, 'r', *SMALL)])
    with pytest.raises(SystemExit):
        main(['train', *run(datadir, tmp_path, 'r', *SMALL)])
    main(['train', *run(datadir, tmp_path, 'r', *SMALL, '--load', '--num-epoch', '4')])
    ckpt = torch.load(tmp_path / 'runs' / 'r' / 'checkpoint.pt', weights_only=False)
    assert ckpt['epoch'] == 4
    assert len((tmp_path / 'runs' / 'r' / 'metrics.csv').read_text().strip().splitlines()) == 1 + 4


def test_yaml_layering_and_old_keys(datadir, tmp_path):
    y1, y2 = tmp_path / 'a.yaml', tmp_path / 'b.yaml'
    y1.write_text('num_channels_m: [[12], [12]]\nnum_channels_2to2: [8, 8]\nnum_channels_m_out: [12, 8]\nnum_channels_out: [8]\n'
                  'num_channels_scalar: 4\nconfig: S\nmethod: spurions\ntask: train\nlr_decay_type: cos\nrank1_width_multiplier: 2\n')
    y2.write_text('config: m\nlr_init: 0.001\nbatchnorm: b\n')
    main(['train', *run(datadir, tmp_path, 'y', '--config', str(y1), '--config', str(y2))])
    import yaml
    saved = yaml.safe_load((tmp_path / 'runs' / 'y' / 'config.yaml').read_text())
    assert saved['agg'] == 'm' and saved['lr_init'] == 0.001 and saved['num_channels_eq'] == [8, 8]
    assert saved['lr_schedule'] == 'cos' and saved['method'] == 'spurions' and saved['batchnorm'] is True


def test_unknown_yaml_key_is_an_error(datadir, tmp_path):
    y = tmp_path / 'bad.yaml'
    y.write_text('lr_innit: 0.1\n')
    with pytest.raises(SystemExit):
        main(['train', *run(datadir, tmp_path, 'bad', '--config', str(y))])


@pytest.mark.parametrize('extra', [['--agg', 'sM'], ['--task', 'regression', '--no-mlp-out'],
                                   ['--irc-safe', '--method', 'input', '--stabilizer', '1', '--agg', 's', '--agg-out', 's'],
                                   ['--num-channels-scalar', '20']])
def test_invalid_configurations_exit_cleanly(datadir, tmp_path, extra):
    with pytest.raises(SystemExit):
        main(['train', *run(datadir, tmp_path, 'inv', *SMALL, *extra)])


def test_info(capsys):
    main(['info', '--num-channels-eq', '8', '8', '--num-channels-m', '12'])
    assert 'parameters' in capsys.readouterr().out


def test_export_predictions(datadir, tmp_path):
    import h5py as _h5
    main(['train', *run(datadir, tmp_path, 'exp', *SMALL)])
    mod = load_script('export_predictions')
    rd = tmp_path / 'runs' / 'exp'
    mod.main([str(rd)])
    pt = torch.load(rd / 'predictions' / 'best_test.pt', weights_only=True)
    with _h5.File(rd / 'predictions' / 'best_test.h5') as f:
        assert np.allclose(f['predict'][:], pt['predict'].numpy()) and np.array_equal(f['targets'][:], pt['targets'].numpy())
    assert mod.export(rd / 'predictions' / 'best_test.pt') is None          # no overwrite without force


@pytest.mark.parametrize('name', ['classifier', 'regressor'])
def test_convert_old_checkpoint(datadir, tmp_path, name):
    """Checkpoints of the pre-package scripts (made with them, see tests/data) reproduce their predictions."""
    import shutil
    from pelican.config import parse_args
    old = tmp_path / 'model' / f'{name}_best.pt'
    old.parent.mkdir()
    shutil.copy(os.path.join(HERE, 'data', f'old_{name}.pt'), old)
    mod = load_script('convert_checkpoint')
    rd = mod.convert(old, runs_dir=str(tmp_path / 'runs'))
    cfg = parse_args(['eval', rd, '--cpu'])
    model = build_model(cfg).double().eval()
    model.load_state_dict(torch.load(os.path.join(rd, 'best.pt'), weights_only=False)['model_state'])
    expected = torch.load(os.path.join(HERE, 'data', 'old_expected.pt'), weights_only=True)[name]
    with torch.no_grad():
        out = model(make_jets(B=6, N=16, seed=7))['predict']
    assert ((out - expected).abs().max() / expected.abs().max()) < 1e-10
    target = ['--target', 'truth_Pmu'] if name == 'regressor' else []
    main(['eval', rd, '--cpu', '--datadir', datadir, *target])
    assert (tmp_path / 'runs' / f'{name}' / 'test_metrics.json').exists()
