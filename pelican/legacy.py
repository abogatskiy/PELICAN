"""
Conversion of checkpoints written by the pre-package PELICAN scripts (train_pelican_classifier.py / train_pelican_cov.py)
to the package's parameter names. Used by scripts/convert_checkpoint.py.
"""
import re

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
