"""
Conversion of checkpoints written by the pre-package PELICAN scripts (train_pelican_classifier.py / train_pelican_cov.py)
to the package's parameter names. Used by scripts/convert_checkpoint.py.
"""
import pickle
import re
import types

import torch

# old module prefix -> new module prefix
_PREFIXES = [(r'^linear\.', 'encoder.'), (r'^input_encoder\.rank2_alphas$', 'encoder.alphas'),
             (r'^net2to2\.message_layers\.', 'body.messages.'), (r'^net2to2\.eq_layers\.', 'body.eqs.'),
             (r'^msg_2to[01]\.', 'message_out.'), (r'^agg_2to[01]\.', 'pool.'), (r'^eq1to2\.', 'lift.'),
             (r'^mlp_out(_1)?\.', 'head.')]


def convert_key(key):
    for pattern, repl in _PREFIXES:
        key = re.sub(pattern, repl, key)
    key = key.replace('.normlayer.', '.norm.')
    # Linear layers of MessageNets (now an nn.Sequential `net` of Linear/activation pairs) and of the output MLP
    m = re.match(r'^(body\.messages\.\d+|message_out)\.linear\.(\d+)\.(weight|bias)$', key)
    if m:
        return f'{m.group(1)}.net.{2 * int(m.group(2))}.{m.group(3)}'
    m = re.match(r'^head\.linear\.(\d+)\.(weight|bias)$', key)
    if m:
        return f'head.{2 * int(m.group(1))}.{m.group(2)}'
    return re.sub(r'\.alphas\.0$', '.alphas', key)


def convert_state_dict(old):
    """Old PELICANClassifier/PELICANRegression state dict -> PelicanModel state dict."""
    new = {}
    for key, value in old.items():
        k = convert_key(key)
        if k.endswith('.alphas') and not k.startswith('encoder'):
            value = value.reshape(value.shape[0], value.shape[1], value.shape[2])     # drop trailing broadcast axes
        elif k == 'pool.bias':
            value = value.reshape(-1)                                               # the old Eq2to0 bias was [1, C]
        new[k] = value
    return new


class _Removed:
    """Stands in for objects of the removed src/ package pickled into old checkpoints (e.g. LR schedulers)."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'src' or module.startswith('src.'):
            return _Removed
        return super().find_class(module, name)


def load_checkpoint(path):
    """torch.load for old checkpoints, which may reference classes of the removed src/ package."""
    return torch.load(path, map_location='cpu', weights_only=False,
                      pickle_module=types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__='pickle'))


def convert_args(old_args, model_state):
    """
    Arguments of an old checkpoint (argparse Namespace or dict) -> {new option: value}, using the same rename/removal
    maps as for old YAML files. Returns (values, ignored old keys).
    """
    from .config import RENAMED, REMOVED, build_parser
    old = dict(vars(old_args)) if not isinstance(old_args, dict) else dict(old_args)
    valid = {a.dest for a in build_parser()._subparsers._group_actions[0].choices['train']._actions}
    values, ignored = {}, []
    for key, value in old.items():
        if key == 'task':                        # old meaning: train | eval
            continue
        if key in REMOVED or RENAMED.get(key, key) not in valid or key == 'yaml':
            ignored.append(key)
            continue
        values[RENAMED.get(key, key)] = value
    if old.get('dropout') is False:
        values['drop_rate'] = values['drop_rate_out'] = 0.0
    if old.get('dtype') == 'double':
        values['double'] = True
    if values.get('dataset') not in ('', 'jc', None):        # e.g. 'jet', the old default
        values['dataset'] = ''
    for key in ('agg', 'agg_out'):
        if key in values and len(values[key]) != 1:
            raise ValueError(f"multi-letter aggregation '{values[key]}' ({key}) is not supported by the package")
    values['task'] = 'classification' if any(k.startswith(('agg_2to0', 'mlp_out.')) for k in model_state) else 'regression'
    values.pop('prefix', None)
    return values, ignored
