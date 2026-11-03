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
def set_experiments_bt_pretrain(n=20, epochs=500, seed=476, path='./hpo/bt_pretrain.json'):
    '''
    n random optimizer/loss samples for each architecture (encoder x proj_dim x init_chan), fixed batch.
    encoder: 'cnn' -> ConditionCNNEncoder, 'vit' -> ConditionViTEncoder.
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


    # ARCHITECTURES (representation dim = 16*init_chan for both encoders)
    grid = []
    for encoder_type in ['cnn', 'vit']:
        for proj_dim in [2048, 4096, 8192]:
            for init_chan in [32, 48]:
                grid.append((encoder_type, proj_dim, init_chan))

    i = 0
    with open(path, 'w') as fp:
        for encoder_type, proj_dim, init_chan in grid:
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


def set_experiments_bt_pretrain_long(bt_id, epochs=1000, path='./hpo/bt_pretrain_long.json', bt_path='./hpo/bt_pretrain.json'):
    '''
    Long BT pretraining of the best encoder from the probe: row bt_id of bt_path, unchanged
    except for epochs. id stays bt_id, so the checkpoint is bt_{bt_id:03}_e{epochs-1:04}.pth.tar
    and fine-tuning rows keep the same bt_id. Launch with:
      ./k8s/kube_launcher.sh bt_pretrain ./hpo/bt_pretrain_long.json <bt_id> <bt_id>
    '''
    with open(bt_path, 'r') as fp:
        bt_rows = {row['id']: row for row in map(json.loads, fp) if row}
    assert bt_id in bt_rows, f"BT id {bt_id} not in {bt_path}"

    exp = dict(bt_rows[bt_id], epochs=epochs)
    with open(path, 'w') as fp:
        fp.write(json.dumps(exp) + '\n')


def set_experiments_bt_train(bt_id, bt_run='bt_pretrain_long', n=40, epochs=100, seed=476, path='./hpo/bt_train.json'):
    '''
    n random fine-tuning samples for the whole BTSegmenter, starting from BT checkpoint bt_id
    of run bt_run (default: the long pretraining of the best encoder from the probe). The BT
    architecture keys are copied from its row in ./hpo/{bt_run}.json so train_condition.py can
    rebuild BarlowTwins and load the checkpoint strictly; bt_run tells k8s/bt_train.sh which
    /model_pvc/<run> folder holds that checkpoint.
    base_lrate is the lr at the reference batch (16); lrate = base_lrate * sqrt(batch/16).
    enc_lrate = lrate * enc_ratio for the pre-trained encoder (1.0 = same lrate as decoder).
    '''
    set_seed(seed)

    # BT ARCHITECTURE OF THE SELECTED ENCODER
    bt_path = f'./hpo/{bt_run}.json'
    with open(bt_path, 'r') as fp:
        bt_rows = {row['id']: row for row in map(json.loads, fp) if row}
    assert bt_id in bt_rows, f"BT id {bt_id} not in {bt_path}"
    bt_arch = {k: bt_rows[bt_id][k] for k in ('init_chan', 'cnn_layers', 'proj_dim', 'encoder')}

    space = {
        'base_lrate': ('log', 5e-5, 1e-3),       # lr at batch 16
        'enc_ratio':  ('choice', [0.1, 0.3, 1.0]),
        'decay':      ('log', 1e-4, 3e-2),       # decoupled AdamW decay; norm weights/biases excluded
        'batch':      ('choice', [16, 32, 48]),
    }

    # FIXED: AdamW (no decay on norm/bias), CE loss, 10-epoch warmup + cosine, grad clip 1.0, bf16, 4 bands, CNN decoder, dihedral+gain aug, best epoch by val IoU_1
    ref_batch = 16
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
            exp.update({'id': i, 'bt_id': bt_id, 'bt_run': bt_run, 'seed': seed, 'epochs': epochs})
            fp.write(json.dumps(exp) + '\n')



################################################################################
# SUPERVISED BASELINE
################################################################################
def set_experiments_base(n=10, epochs=100, seed=476, path='./hpo/base_train.json'):
    '''
    Supervised BaseSegmenter (train_base.py) trained from scratch. Grid over architecture and batch,
    n random (lrate, decay) samples per grid cell:
      encoder x decoder: cnn-cnn, cnn-vit, vit-cnn, vit-vit
      init_chan:         32, 64
      vit_layers:        1, 2 (ViT layers per ViT stage; 0 for cnn-cnn, which has no ViT stage)
      batch:             16, 32 (48 dropped to limit rows)
    -> (2 + 3*2*2) archs x 2 batches x n = 280 rows at n=10, grouped by arch then batch.
    base_lrate is the lr at the reference batch (16); lrate = base_lrate * sqrt(batch/16).
    Launch with:
      ./k8s/kube_launcher.sh base_train ./hpo/base_train.json 0 <rows-1>
    '''
    set_seed(seed)

    space = {
        'base_lrate': ('log', 5e-5, 1e-3),       # lr at batch 16
        'decay':      ('log', 1e-4, 1e-1),       # decoupled AdamW decay; norm weights/biases excluded
    }

    # FIXED: AdamW (no decay on norm/bias), CE loss, 10-epoch warmup + cosine, grad clip 1.0, bf16, 4 bands, dihedral+gain aug, best epoch by val IoU_1
    ref_batch  = 16
    cnn_layers = 2
    batches    = [16, 32]

    # ARCHITECTURES (cnn-cnn has no ViT stage, so no vit_layers sweep)
    grid = []
    for encoder_type, decoder_type in itertools.product(['cnn', 'vit'], ['cnn', 'vit']):
        vit_layers = [0] if encoder_type == decoder_type == 'cnn' else [1, 2]
        for init_chan, n_vit in itertools.product([32, 64], vit_layers):
            grid.append((encoder_type, decoder_type, init_chan, n_vit))

    i = 0
    with open(path, 'w') as fp:
        for (encoder_type, decoder_type, init_chan, n_vit), batch in itertools.product(grid, batches):
            for _ in range(n):

                exp = {k: sample(v) for k, v in space.items()}
                exp['batch']      = batch
                exp['lrate']      = exp['base_lrate'] * (batch / ref_batch) ** 0.5 #lrate adjust rule
                exp['encoder']    = encoder_type
                exp['decoder']    = decoder_type
                exp['init_chan']  = init_chan
                exp['cnn_layers'] = cnn_layers
                exp['vit_layers'] = n_vit

                # ROUND (3 SIGNIFICANT FIGURES)
                for k in ('base_lrate', 'lrate', 'decay'):
                    exp[k] = float(f"{exp[k]:.3g}")

                exp.update({'id': i, 'seed': seed, 'epochs': epochs})
                fp.write(json.dumps(exp) + '\n')
                i += 1


################################################################################
# DIFFUSION
################################################################################
def set_experiments_diff_train(n=30,epochs=500,seed=476,path='./hpo/diff_train.json'):
    '''
    Unguided diffusion models. List of experiments searches hyperparameter space for
    a semantic segmentation model with pre-trained weights. Pre-trained weights are
    transfered from an unguided diffusion model.
    '''
    set_seed(seed)
    return


def set_experiments_diff_pretrain(n=30,epochs=500,seed=476,path='./hpo/diff_pretrain.json'):
    '''
    Unguided diffusion models. List of experiments searches the hyperparameter space
    among these.
    '''
    set_seed(seed)
    return

################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
    set_experiments_base()
