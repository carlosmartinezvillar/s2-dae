import itertools
import json
import os
import random
import numpy as np


def sample(spec):
    '''
    ('log',lo,hi)   -> log-uniform in [lo,hi]
    ('choice',[..]) -> uniform over the list
    '''
    if spec[0] == 'log':
        _, lo, hi = spec
        return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
    return spec[1][np.random.randint(len(spec[1]))]


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)


################################################################################
# BARLOW TWINS
################################################################################
def set_experiments_bt_pretrain(n=30, epochs=500, seed=476, path='./hpo/bt_pretrain.json'):
    '''
    n random optimizer/loss samples for each architecture (proj_dim x init_chan), fixed batch.
    base_lrate is the lr at the reference batch (256). AdamW square-root scaling:
    lrate = base_lrate * sqrt(batch/256), so base_lrate stays comparable if batch is changed later.
    '''
    set_seed(seed)

    space = {
        'base_lrate': ('log', 1e-4, 2e-3),       # lr at batch 256 (lrate == base_lrate at batch 256)
        'decay':      ('log', 1e-3, 1e-1),       # decoupled AdamW decay; norm weights/biases excluded
        'lambda_d':   ('log', 1.0, 40.0),        # bt_lambda * proj_dim (off-diag terms grow ~D^2)
    }


    # FIXED
    batch = 256
    ref_batch = 256
    cnn_layers = 2
    encoder_type = 'cnn'


    # ARCHITECTURES (representation dim = 16*init_chan)
    grid = []
    for proj_dim in [2048, 4096, 8192]:
        for init_chan in [32, 48]:
            grid.append((proj_dim, init_chan))

    i = 0
    with open(path, 'w') as fp:
        for proj_dim, init_chan in grid:
            for _ in range(n):

                exp = {k: sample(v) for k, v in space.items()}
                exp['proj_dim']   = proj_dim
                exp['init_chan']  = init_chan
                exp['cnn_layers'] = cnn_layers
                exp['batch']      = batch  # images per batch
                exp['lrate']      = exp['base_lrate'] * (batch / ref_batch) ** 0.5 #lrate adjust rule
                exp['encoder']    = encoder_type

                # ROUND (3 SIGNIFICANT FIGURES)
                for k in ('base_lrate', 'lrate', 'decay', 'lambda_d'):
                    exp[k] = float(f"{exp[k]:.3g}")
                exp['bt_lambda'] = float(f"{exp['lambda_d'] / proj_dim:.3g}")

                exp.update({'id': i, 'seed': seed, 'epochs': epochs})
                fp.write(json.dumps(exp) + '\n')
                i += 1


def set_experiments_bt_train(bt_id, n=30, epochs=100, seed=476, path='./hpo/bt_whole.json', bt_path='./hpo/bt_pretrain.json'):
    '''
    n random fine-tuning samples for the whole BTSegmenter, starting from BT checkpoint bt_id
    (the best encoder from the probe). The BT architecture keys are copied from its row in
    bt_path so validate_condition.py can rebuild BarlowTwins and load the checkpoint strictly.
    base_lrate is the lr at the reference batch (32); lrate = base_lrate * sqrt(batch/32).
    enc_lrate = lrate * enc_ratio for the pre-trained encoder (1.0 = same lrate as decoder).
    '''
    set_seed(seed)

    # BT ARCHITECTURE OF THE SELECTED ENCODER
    with open(bt_path, 'r') as fp:
        bt_rows = {row['id']: row for row in map(json.loads, fp) if row}
    assert bt_id in bt_rows, f"BT id {bt_id} not in {bt_path}"
    bt_arch = {k: bt_rows[bt_id][k] for k in ('init_chan', 'cnn_layers', 'proj_dim', 'encoder')}

    space = {
        'base_lrate': ('log', 1e-4, 3e-3),       # lr at batch 32
        'enc_ratio':  ('choice', [0.1, 0.3, 1.0]),
        'decay':      ('log', 1e-4, 1e-1),       # decoupled AdamW decay; norm weights/biases excluded
        'batch':      ('choice', [16, 32, 64]),
    }

    # FIXED
    ref_batch = 32
    decoder_type = 'cnn' # BTSegmenter only has BaseCNNSegDecoder for now

    with open(path, 'w') as fp:
        for i in range(n):

            exp = {k: sample(v) for k, v in space.items()}
            exp['lrate']     = exp['base_lrate'] * (exp['batch'] / ref_batch) ** 0.5 #lrate adjust rule
            exp['enc_lrate'] = exp['lrate'] * exp['enc_ratio']
            exp['decoder']   = decoder_type

            # ROUND (3 SIGNIFICANT FIGURES)
            for k in ('base_lrate', 'lrate', 'enc_lrate', 'decay'):
                exp[k] = float(f"{exp[k]:.3g}")

            exp.update(bt_arch)
            exp.update({'id': i, 'bt_id': bt_id, 'seed': seed, 'epochs': epochs})
            fp.write(json.dumps(exp) + '\n')



################################################################################
# DIFFUSION
################################################################################
def set_experiments_unguided_train(n=30,epochs=500,seed=476,path='./hpo/unguided.json'):
	set_seed(seed)
	return


def set_experiments_unguided(n=30,epochs=500,seed=476,path='./hpo/unguided.json'):
	set_seed(seed)
	return

################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	set_experiments_bt_pretrain()
