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



def set_experiments_bt_pretrain_final(n=20, epochs=500, seed=476, path='./hpo/bt_pretrain_final.json'):
    '''
    Second BT sweep, zoomed in on the ranges that worked in the 240-run bt_pretrain sweep (bt_probe val iou_1),
    over the same encoder grid as base_train so the encoders stay comparable to the supervised baseline:
      cnn: init_chan 32, 64 (vit_layers 0)
      vit: init_chan 32, 64 x vit_layers 1, 2 (ViT layers per ViT stage; the bt_pretrain sweep only had 1)
    -> 6 cells x n = 120 rows at n=20, ids from 0, grouped by encoder, then vit_layers, then init_chan.
    Each cell samples its own lrate/decay/lambda_d, so depth and width each get a tuned lrate.
    Ranges per encoder, from bt_pretrain (Spearman rho with probe iou_1 over 120 runs per encoder, |rho| > 0.18 ~ p < 0.05):
      cnn base_lrate 5e-4..3e-3   rho +0.59; probe rises up to ~6e-4, plateau 6e-4..1.9e-3 (sampled max), best 076 at 1.3e-3
          decay      5e-4..3e-2   rho -0.25; medians drop above ~1.6e-2, best 076 at 2.5e-2; lowest bin was the sampled min
          lambda_d   0.5..12      rho -0.30; best median in the lowest bin (sampled min 1.0); top 20% mostly 1.4..12
      vit base_lrate 5e-5..1e-3   rho 0.00 (flat); best 121/193 at 1.2e-4/2.2e-4; extended down for vit_layers 2
                                  (deeper ViT encoders preferred ~1.7x lower lrate in base_train)
          decay      2e-3..1e-1   rho -0.05 (flat); top 20% 2.4e-3..7e-2
          lambda_d   0.5..20      rho -0.25; best median in the lowest bin; 121/193 at 1.1/17.4
    proj_dim is fixed at 4096: no effect on the probe (cnn rho +0.01, top-5 means 0.9195/0.9211/0.9191 at
    2048/4096/8192; vit rho +0.15, 0.9120/0.9125/0.9119), and 076/193 used it. bt_lambda = lambda_d / proj_dim.
    Fixed as in bt_pretrain: batch 256 (lrate = base_lrate), cnn_layers 2, 500 epochs.
    Rows carry bt_run so the probe finds the checkpoints (hpo/bt_probe_final.json -> bt_pretrain_final.json,
    launched with bt_probe); downstream runs (bt_train, diff_pretrain) take the best rows of this run.
    Measured peaks at batch 256 (vit_layers 1): cnn 32 -> 64 ~17 -> 32 GB, vit 32 -> 64 21 -> 42 GB; vit_layers 2
    adds ~1.45x activations (~61 GB at 64), within an 80 GB A100.
    Launch with:
      ./k8s/kube_launcher.sh bt_pretrain ./hpo/bt_pretrain_final.json 0 <rows-1>
    '''
    set_seed(seed)

    spaces = {
        'cnn': {
            'base_lrate': ('log', 5e-4, 3e-3),   # lr at batch 256 (lrate == base_lrate at batch 256)
            'decay':      ('log', 5e-4, 3e-2),   # decoupled AdamW decay; norm weights/biases excluded
            'lambda_d':   ('log', 0.5, 12.0),    # bt_lambda * proj_dim (off-diag terms grow ~D^2)
        },
        'vit': {
            'base_lrate': ('log', 5e-5, 1e-3),
            'decay':      ('log', 2e-3, 1e-1),
            'lambda_d':   ('log', 0.5, 20.0),
        },
    }

    # FIXED
    batch      = 256
    ref_batch  = 256
    cnn_layers = 2
    proj_dim   = 4096

    # ARCHITECTURES (cnn has no ViT stage, so no vit_layers sweep)
    grid = []
    for encoder_type in ['cnn', 'vit']:
        vit_layers = [0] if encoder_type == 'cnn' else [1, 2]
        for n_vit, init_chan in itertools.product(vit_layers, [32, 64]):
            grid.append((encoder_type, n_vit, init_chan))

    i = 0
    with open(path, 'w') as fp:
        for encoder_type, n_vit, init_chan in grid:
            for _ in range(n):

                exp = {k: sample(v) for k, v in spaces[encoder_type].items()}
                exp['proj_dim']   = proj_dim
                exp['init_chan']  = init_chan
                exp['cnn_layers'] = cnn_layers
                exp['vit_layers'] = n_vit
                exp['batch']      = batch  # images per batch
                exp['lrate']      = exp['base_lrate'] * (batch / ref_batch) ** 0.5 #lrate adjust rule
                exp['encoder']    = encoder_type

                # ROUND (3 SIGNIFICANT FIGURES)
                for k in ('base_lrate', 'lrate', 'decay', 'lambda_d'):
                    exp[k] = float(f"{exp[k]:.3g}")
                exp['bt_lambda'] = float(f"{exp['lambda_d'] / proj_dim:.3g}")

                exp.update({'id': i, 'seed': seed, 'epochs': epochs, 'bt_run': 'bt_pretrain_final'})
                fp.write(json.dumps(exp) + '\n')
                i += 1



def set_experiments_bt_train(bt_ids=(76, 193), decoders=('cnn', 'vit'), enc_ratios=(0.1, 0.5, 1.0), batch=32, bt_run='bt_pretrain', n=30, epochs=100, seed=476, path='./hpo/bt_train.json'):
    '''
    Fine-tuning of the whole BTSegmenter (encoder + decoder, train_condition.py without --probe-only).
    Grid over BT encoder x decoder x enc_ratio, n random (base_lrate, decay) samples per cell:
      bt_ids:     BT checkpoints of run bt_run, default the best bt_probe CNN 076 and ViT 193 (500 epochs)
      decoders:   'cnn' (BaseCNNSegDecoder), 'vit' (BaseViTSegDecoder)
      enc_ratios: encoder lrate as a fraction of the decoder's, enc_lrate = lrate * enc_ratio
    -> 2 x 2 x 3 x n = 360 rows at n=30, grouped by bt_id, then decoder, then enc_ratio.
    batch is fixed (32, the probe's): batch 16 and 32 tied in base_train, and the sqrt rule below absorbs it.
    The BT architecture keys are copied from the bt_id row of ./hpo/{bt_run}.json so train_condition.py
    can rebuild BarlowTwins and load the checkpoint strictly; bt_run tells k8s/bt_train.sh which
    /model_pvc/<run> folder holds that checkpoint. Width is the checkpoint's init_chan (48 for both
    defaults); the decoder follows it. vit_layers (ViT layers per ViT stage) is shared by encoder and decoder,
    as in BaseSegmenter: a ViT encoder's comes from its BT row (1 if the row has none, the bt_pretrain
    checkpoints) and a ViT decoder uses the same; with a CNN encoder it is 0 for a CNN decoder, 1 for a ViT one.
    base_lrate is the lr at the reference batch (16); lrate = base_lrate * sqrt(batch/16).
    enc_lrate = lrate * enc_ratio for the pre-trained encoder (1.0 = same lrate as decoder).
    Launch with:
      ./k8s/kube_launcher.sh bt_train ./hpo/bt_train.json 0 <rows-1>
    '''
    set_seed(seed)

    # BT ARCHITECTURE OF THE SELECTED ENCODERS
    bt_path = f'./hpo/{bt_run}.json'
    with open(bt_path, 'r') as fp:
        bt_rows = {row['id']: row for row in map(json.loads, fp) if row}

    space = {
        'base_lrate': ('log', 5e-5, 3e-4),       # lr at batch 16 (lrate <= 4.2e-4 at batch 32); base_train: plateau below ~5e-4 lrate, every pair with a ViT part collapses above ~5e-4-8e-4; probe 2e-4 @ batch 32 = 1.4e-4
        'decay':      ('log', 1e-4, 3e-2),       # decoupled AdamW decay; norm weights/biases excluded
    }

    # FIXED: AdamW (no decay on norm/bias), CE loss, 10-epoch warmup + cosine, grad clip 1.0, bf16, 4 bands, dihedral+gain aug, best epoch by val IoU_1
    ref_batch = 16

    i = 0
    with open(path, 'w') as fp:
        for bt_id, decoder_type, enc_ratio in itertools.product(bt_ids, decoders, enc_ratios):
            assert bt_id in bt_rows, f"BT id {bt_id} not in {bt_path}"
            assert decoder_type in ('cnn', 'vit'), f"Unknown decoder '{decoder_type}'"
            bt_arch = {k: bt_rows[bt_id][k] for k in ('init_chan', 'cnn_layers', 'proj_dim', 'encoder')}

            for _ in range(n):

                exp = {k: sample(v) for k, v in space.items()}
                exp['enc_ratio']  = enc_ratio
                exp['batch']      = batch
                exp['lrate']      = exp['base_lrate'] * (batch / ref_batch) ** 0.5 #lrate adjust rule
                exp['enc_lrate']  = exp['lrate'] * enc_ratio
                exp['decoder']    = decoder_type
                if bt_arch['encoder'] == 'vit':
                    exp['vit_layers'] = bt_rows[bt_id].get('vit_layers', 1) # MUST MATCH THE CHECKPOINT (STRICT LOAD)
                else:
                    exp['vit_layers'] = 0 if decoder_type == 'cnn' else 1

                # ROUND (3 SIGNIFICANT FIGURES)
                for k in ('base_lrate', 'lrate', 'enc_lrate', 'decay'):
                    exp[k] = float(f"{exp[k]:.3g}")

                exp.update(bt_arch)
                exp.update({'id': i, 'bt_id': bt_id, 'bt_run': bt_run, 'seed': seed, 'epochs': epochs})
                fp.write(json.dumps(exp) + '\n')
                i += 1



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



def set_experiments_diff_pretrain(bt_id=76, bt_run='bt_pretrain', n=5, epochs=200, seed=476, path='./hpo/diff_pretrain.json'):
    '''
    Unguided denoiser (models.UnconditionedDenoiser, pretrain_denoiser.py) trained on the 4 VNIR bands.
    Grid over architecture, n random (lrate, decay) samples per grid cell:
      arch:       cnn (5 CNN stages), vit (2 CNN + 3 ViT stages)
      init_chan:  32, 64
      vit_layers: 1, 2 (ViT layers per ViT stage; 0 for cnn)
    -> (2 + 2*2) archs x n = 30 rows at n=5, grouped by arch.
    epochs=200 is the sweep length (~460 steps/epoch at batch 32 on 14,682 chips -> ~92k steps); rerun
    the best row longer (500-1000 epochs) once the KID curves are in.
    bt_id/bt_run: the frozen BT encoder used by the metrics (fid, kid, lpips), the same for every row so
    scores are comparable. bt_id is a row id of ./hpo/{bt_run}.json (default: CNN 076 of the bt_pretrain
    sweep, the best bt_probe encoder, 500 epochs, C = 16*48 = 768). k8s/diff_pretrain.sh copies bt_{bt_id:03}_e*.pth.tar from
    /model_pvc/<bt_run> and pretrain_denoiser.py reads the architecture from the same row.
    bt_id None writes a placeholder, which the pod and the script refuse to run.
    base_lrate is the lr at the reference batch (16); lrate = base_lrate * sqrt(batch/16).
    Launch with:
      ./k8s/kube_launcher.sh diff_pretrain ./hpo/diff_pretrain.json 0 <rows-1>
    '''
    set_seed(seed)

    # THE METRIC ENCODER MUST BE A ROW OF THE BT RUN. None -> PLACEHOLDER
    if bt_id is None:
        print(f"WARNING: bt_id is None in {path} -- set it to the chosen row of ./hpo/{bt_run}.json before launching")
    else:
        bt_path = f'./hpo/{bt_run}.json'
        with open(bt_path, 'r') as fp:
            bt_ids = {row['id'] for row in map(json.loads, fp) if row}
        assert bt_id in bt_ids, f"BT id {bt_id} not in {bt_path}"

    space = {
        'base_lrate': ('log', 5e-5, 5e-4),       # lr at batch 16
        'decay':      ('log', 1e-4, 1e-2),       # decoupled AdamW decay; norm weights/biases excluded
    }

    # FIXED: AdamW (no decay on norm/bias), noise-prediction MSE, 10-epoch warmup + cosine, grad clip 1.0,
    # bf16, 4 bands, dihedral aug (no gain), 1000-step cosine noise schedule, best checkpoint by KID
    ref_batch  = 16
    batch      = 32
    cnn_layers = 2

    # ARCHITECTURES (cnn has no ViT stage, so no vit_layers sweep)
    grid = []
    for arch in ['cnn', 'vit']:
        vit_layers = [0] if arch == 'cnn' else [1, 2]
        for init_chan, n_vit in itertools.product([32, 64], vit_layers):
            grid.append((arch, init_chan, n_vit))

    i = 0
    with open(path, 'w') as fp:
        for arch, init_chan, n_vit in grid:
            for _ in range(n):

                exp = {k: sample(v) for k, v in space.items()}
                exp['batch']      = batch
                exp['lrate']      = exp['base_lrate'] * (batch / ref_batch) ** 0.5 #lrate adjust rule
                exp['arch']       = arch
                exp['init_chan']  = init_chan
                exp['cnn_layers'] = cnn_layers
                exp['vit_layers'] = n_vit

                # ROUND (3 SIGNIFICANT FIGURES)
                for k in ('base_lrate', 'lrate', 'decay'):
                    exp[k] = float(f"{exp[k]:.3g}")

                exp.update({'id': i, 'bt_id': bt_id, 'bt_run': bt_run, 'seed': seed, 'epochs': epochs})
                fp.write(json.dumps(exp) + '\n')
                i += 1

################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
    # SECOND BT SWEEP, ZOOMED IN ON bt_pretrain; cnn/vit x init_chan 32/64 (x vit_layers 1/2), LIKE base_train
    set_experiments_bt_pretrain_final()

    # METRIC ENCODER FOR THE DIFFUSION SCORES: CNN 076 OF THE bt_pretrain SWEEP (BEST bt_probe)
    set_experiments_diff_pretrain(bt_id=76, bt_run='bt_pretrain')
