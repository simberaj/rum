
import itertools


INIT = 'python train_model.py tallinn rfor tgt_eegrid -s 42'
# INIT = 'python train_model.py prague rfor tgt_obyv_adrpt -s 42'
OUT_PATH = 'models\\for_maribor_tal\\'

X = [
# 'buildheight',
'building_diss',
'building(?!_dis{2})',
'poi',
'srtm',
'transport',
'urbanatlas',
]

def outname(parts, allparts):
    if len(parts) < len(allparts) / 2:
        return ''.join(c for c in ''.join(parts) if c not in '()!?_{}')
    else:
        return 'no' + ''.join(c for c in ''.join(part for part in allparts if part not in parts) if c not in '()!?_{}')

cmds = []

for inds in itertools.product((False, True), repeat=len(X)):
    parts = [x for x, i in zip(X, inds) if i]
    # if len(parts) in [1, len(X)-1]:
    if len(parts) in [len(X)-1]:
        cmds.append(' '.join([
            INIT,
            '-r',
            '"(' + '.*)|('.join(parts) + '.*)"',
            OUT_PATH + outname(parts, X) + '.rum',
        ]))


open('runtrain.bat', 'w').write('\n'.join(cmds))