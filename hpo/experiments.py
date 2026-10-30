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
# BEST SUPERVISED BASELINES -- s2dw-lakes-dataset (models/results.py), 3 BANDS (RGB)
################################################################################
# Stage 1: 40 random trials/model (lrate, decay, batch 8|16, 35 epochs); stage 2: top 3 x 5 seeds;
# stage 3 picks (ids 000,055,018,065): AdamW (decay on all params), CE, 5-epoch warmup + cosine to 0, 65 epochs.
# Stage 4 test IoU (%) Tiny/Small/Base/Large -- CNN-CNN 89.14/89.73/90.06/89.76 | ViT-CNN 89.36/89.85/88.89/89.82
#   | CNN-ViT 89.81/90.46/90.42/90.76 | ViT-ViT 89.74/90.57/90.63/90.94 (best overall: ViT-ViT Large).
# Sizes (channels,cnn_layers,vit_layers): Tiny (16,2,1), Small (16,3,2), Base (32,2,1), Large (32,3,2);
#   stage 4 trained all sizes for 100 epochs with lrate 2e-4 and each model's decay rounded (8e-3, 5e-3, 8e-3, 1e-2).
LAKES_BEST = {
    'UNet_CNN_CNN': {'lrate': 2.2e-4, 'decay': 8.37e-3, 'batch': 16, 'channels': 32, 'cnn_layers': 2, 'vit_layers': 1, 'mlp_ratio': 5, 'epochs': 65},
    'UNet_ViT_CNN': {'lrate': 2.0e-4, 'decay': 5.0e-3,  'batch': 16, 'channels': 32, 'cnn_layers': 2, 'vit_layers': 1, 'mlp_ratio': 5, 'epochs': 65},
    'UNet_CNN_ViT': {'lrate': 2.0e-4, 'decay': 7.86e-3, 'batch': 16, 'channels': 32, 'cnn_layers': 2, 'vit_layers': 1, 'mlp_ratio': 5, 'epochs': 65},
    'UNet_ViT_ViT': {'lrate': 2.0e-4, 'decay': 1.0e-2,  'batch': 16, 'channels': 32, 'cnn_layers': 2, 'vit_layers': 1, 'mlp_ratio': 5, 'epochs': 65},
}


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
    Ranges are centred on the supervised optimum in LAKES_BEST (lrate ~2e-4, decay 5e-3..1e-2 at batch 16);
    n=40 matches the 40 stage-1 trials per model the baselines got.
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
