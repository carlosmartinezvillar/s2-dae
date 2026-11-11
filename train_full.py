import os
import glob
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import time
import argparse
# import json
# import inspect

import dataloader
import utils
import models
from diffusion import GaussianDiffusion, mask_to_x0, x0_to_mask, mask_channels

# FIXED VALIDATION RNG -- SAME t AND NOISE EVERY EPOCH, SO VALIDATION MSE IS COMPARABLE ACROSS EPOCHS
VAL_SEED = 0


def parse_args():
	'''
	Load args.
	'''
	# DEFINITION
	parser = argparse.ArgumentParser()
	required = parser.add_argument_group('Required arguments')
	required.add_argument('--data-dir',required=True,help='Directory of .npy cached datasets')
	required.add_argument('--net-dir',required=True,help='Output dir for trained model weights.')
	required.add_argument('--log-dir',required=True,help='Output dir for training logs.')
	required.add_argument('--params',required=True,help='JSON hyperparameter file.')
	required.add_argument('--id',required=True,type=int,help='model id in hyperparameter file.')

	# OPTIONAL
	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--num-workers',required=False,type=int,default=4,help='Set num_workers args.')
	optional.add_argument('--device',required=False,type=int,default=0,help='Override default GPU 0')

	# LOAD
	args = parser.parse_args ()

	# CHECK REQUIRED
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"

	# CHECK OPTIONAL
	assert args.device >= 0, f"Got negative arg for GPU id {args.device}"
	if args.device > 0:
		assert args.device < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."	
	# assert args.workers in range(0,9), "Got out of range arg for nr of workers"

	# SET GLOBAL VARS -- skip passing as args in train_probe(), train_whole(), etc.
	global LOG_DIR
	global MODEL_DIR
	LOG_DIR = args.log_dir
	MODEL_DIR = args.net_dir

	# RETURN
	return args


def load_cond_encoder(model,hp,epoch=None):
	'''
	Pre-trained BT encoder -> model.cond_encoder, strict.
	Reads {MODEL_DIR}/bt_{bt_id:03}_e{epoch:04}.pth.tar, latest epoch if epoch is None (k8s/full_train.sh copies
	it from /model_pvc/<bt_run>). bt_id is hp['bt_id'] if set, else hp['id'] (same rule as full_train.sh).
	The model must be built with the BT row's architecture (cond_encoder = its encoder, init_chan_cond = its
	init_chan, cond_cnn_layers = its cnn_layers); the strict load catches any mismatch.
	'''
	bt_id = hp.get('bt_id',hp['id'])

	# FIND CHECKPOINT
	epoch_str = '*' if epoch is None else f'{epoch:04}'
	paths = sorted(glob.glob(f"{MODEL_DIR}/bt_{bt_id:03}_e{epoch_str}.pth.tar"))
	assert paths, f"No BT checkpoint bt_{bt_id:03}_e{epoch_str}.pth.tar in {MODEL_DIR}"
	path = paths[-1] # ZERO-PADDED EPOCHS, SO LAST IS LATEST
	checkpoint = torch.load(path,map_location='cpu',weights_only=True)

	# KEEP THE ENCODER, DROP THE PROJECTOR: 'encoder.<name>' -> '<name>'
	state = {k[len('encoder.'):]:v for k,v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
	model.cond_encoder.load_state_dict(state,strict=True)
	print(f"Loaded condition encoder from {path} (epoch {checkpoint['epoch']})")


def load_pretrained_denoiser(model,hp,epoch=None):
	'''
	Pre-trained unguided denoiser (diff_pretrain, models.UnconditionedDenoiser) -> the denoiser of model.
	Reads {MODEL_DIR}/diff_{diff_id:03}_e{epoch:04}.pth.tar, latest epoch if epoch is None (k8s/full_train.sh copies
	it from /model_pvc/diff_pretrain). diff_id is hp['diff_id'] if set, else hp['id'] (same rule as full_train.sh).
	in_layer/out_layer are dropped (bands in/out there, mask in/out here) and the load is non-strict; the only
	keys allowed to stay missing are in_layer, out_layer, cond_encoder and the zero-init context_proj, so a
	checkpoint with a different arch, cnn_layers, vit_layers or init_chan fails here instead of loading partially.
	'''
	diff_id = hp.get('diff_id',hp['id'])

	# FIND CHECKPOINT
	epoch_str = '*' if epoch is None else f'{epoch:04}'
	paths = sorted(glob.glob(f"{MODEL_DIR}/diff_{diff_id:03}_e{epoch_str}.pth.tar"))
	assert paths, f"No denoiser checkpoint diff_{diff_id:03}_e{epoch_str}.pth.tar in {MODEL_DIR}"
	path = paths[-1] # ZERO-PADDED EPOCHS, SO LAST IS LATEST
	checkpoint = torch.load(path,map_location='cpu',weights_only=True)

	# DROP THE BAND-SPECIFIC INPUT/OUTPUT LAYERS, LOAD THE REST
	state  = {k:v for k,v in checkpoint['model_state_dict'].items() if not k.startswith(('in_layer.','out_layer.'))}
	result = model.load_state_dict(state,strict=False)

	# NON-STRICT, SO CHECK EXACTLY WHAT DID NOT LOAD
	assert not result.unexpected_keys, f"Unexpected keys from {path}: {result.unexpected_keys}"
	not_expected = [k for k in result.missing_keys if not (k.startswith(('in_layer.','out_layer.','cond_encoder.')) or 'context_proj' in k)]
	assert not not_expected, f"Keys missing from {path} (architecture mismatch?): {not_expected}"
	print(f"Loaded denoiser from {path} (epoch {checkpoint['epoch']}, {len(state)} tensors)")


def calculate_cross_entropy(x0_pred, true_mask, eps=1e-6):
	'''
	Pixel-wise cross-entropy of the sampled x0 against the true mask.
	x0_pred: [B,C,H,W] in [-1,1] (see mask_to_x0); true_mask: [B,H,W] integer class map.
	x0_pred is mapped back to [0,1]. If C == 1 it is the foreground probability (binary CE);
	otherwise its log is passed as logits, so cross_entropy's softmax renormalizes it over
	classes into per-pixel probabilities.
	Returns a 0-d tensor on x0_pred's device (no .item(), so no GPU-CPU sync).
	'''
	if x0_pred.shape[1] == 1:
		probs = ((x0_pred[:,0] + 1) / 2).clamp(eps,1-eps)
		return F.binary_cross_entropy(probs, true_mask.float())
	probs = ((x0_pred + 1) / 2).clamp(min=eps)
	return F.cross_entropy(torch.log(probs), true_mask.long())

################################################################################
# TRAIN/VALIDATE
################################################################################
def train(model,gpu_aug,diffusion,loader,optimizer,scheduler,device,n_classes=2):

	model.train()

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	for vnir,mask in loader:

		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)
		vnir,mask = gpu_aug(vnir,mask)

		x0  = mask_to_x0(mask,n_classes)
		t   = torch.randint(0,diffusion.timesteps,(vnir.shape[0],),device=device,dtype=torch.long)

		# FORWARD -- NOISE x0 (MASK) TO x_t, PREDICT THE NOISE CONDITIONED ON THE BANDS
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):

			noise = torch.randn_like(x0)

			x_t = diffusion.q_sample(x0,t,noise)

			predicted_noise = model(x_t,t,vnir).float()

			loss = F.mse_loss(predicted_noise,noise)

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
		optimizer.step()


		loss_sum  += loss.detach() * vnir.size(0)
		n_samples += vnir.size(0)

	scheduler.step()

	return (loss_sum/n_samples).item() # >>>> CPU-GPU SYNC <<<<


@torch.no_grad()
def validate(model,gpu_norm,diffusion,loader,device,n_classes=2):
	'''
	Noise-prediction (MSE) loss over the full validation set. t and noise are drawn
	from a fixed seed, so every epoch is scored on the same (x0,t,noise) triples.
	'''

	model.eval()

	mse_loss_sum = torch.zeros(1,device=device)
	n_samples    = torch.zeros(1,device=device)

	# FORK THE RNG SO THE FIXED SEED DOES NOT RESET THE TRAINING RNG STREAM
	with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
		torch.manual_seed(VAL_SEED)

		for vnir,mask in loader:

			vnir = vnir.to(device,non_blocking=True)
			mask = mask.to(device,non_blocking=True)
			vnir,mask = gpu_norm(vnir,mask)

			x0 = mask_to_x0(mask,n_classes)
			t  = torch.randint(0,diffusion.timesteps,(vnir.shape[0],),device=device,dtype=torch.long)

			# FORWARD -- NOISE x0 (MASK) TO x_t, PREDICT THE NOISE CONDITIONED ON THE BANDS
			with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):

				noise = torch.randn_like(x0)

				x_t = diffusion.q_sample(x0,t,noise)

				predicted_noise = model(x_t,t,vnir).float()

				mse_loss = F.mse_loss(predicted_noise,noise)

			mse_loss_sum += mse_loss.detach() * vnir.size(0)
			n_samples    += vnir.size(0)

	return (mse_loss_sum/n_samples).item()

################################################################################
# SAMPLE & SCORE/VALIDATE
################################################################################
@torch.no_grad()
def validate_with_sampling(model,gpu_norm,diffusion,dataloader,device,n_classes=2,seed=0):
	'''
	Segmentation metrics from DDIM-sampled masks, on a fixed subset of the validation set.
	The sampling noise uses a fixed seed on a forked RNG, so every evaluation sees the
	same noise (comparable across epochs/runs) and the training RNG stream is untouched.
	'''
	model.eval()

	ce_loss_sum = torch.zeros(1,device=device)
	n_samples   = torch.zeros(1,device=device)
	gpu_cmat    = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)

	with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
		torch.manual_seed(seed)

		for vnir,mask in dataloader:

			vnir = vnir.to(device,non_blocking=True)
			mask = mask.to(device,non_blocking=True)			
			vnir,mask = gpu_norm(vnir,mask)

			with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
				# STARTING NOISE x_T IN MASK SPACE [B,mask_channels,H,W]
				x_t = torch.randn((vnir.shape[0],mask_channels(n_classes),*vnir.shape[-2:]),device=device)

				# x0_pred = diffusion.p_sample_loop(x_t,vnir)
				x0_pred = diffusion.ddim_sample_loop(x_t,vnir)
			pred_mask = x0_to_mask(x0_pred)

			ce_loss_sum += calculate_cross_entropy(x0_pred.float(),mask) * vnir.size(0)
			n_samples   += vnir.size(0)

			utils.update_confusion_matrix(gpu_cmat,mask,pred_mask,n_classes)

	# CPU-GPU SYNC
	closs    = (ce_loss_sum/n_samples).item() 
	cpu_cmat = gpu_cmat.cpu()

	result = utils.calculate_seg_metrics(cpu_cmat)
	flat_result = {f"{k}_{i}":result[k][i].item() for k in result.keys() for i in range(n_classes)}
	flat_result.update({'closs': closs})

	return flat_result

################################################################################
# RUN
################################################################################
def train_and_validate(model,gpu_aug,gpu_norm,dataloaders,sample_dloader,optimizer,scheduler,diffusion,device,n_classes=2,epochs=100):
	'''
	Train, validate (MSE), score DDIM-sampled masks on sample_dloader every 5 epochs, log, and keep the best
	checkpoints by sampled iou_1 (only scored epochs after warmup can be saved).
	'''

	# LOG PATHS/BUFFERS
	logger  = utils.DiffLogger(f'{LOG_DIR}/full_{model.model_id:03}.tsv')
	tracker = utils.RecentBestTracker(n=3)
	best_iou = 0.0
	best_iou_epoch = 0

	# SAMPLED COLUMNS, SAME ORDER AS validate_with_sampling(); NaN ON EPOCHS WITHOUT SAMPLING.
	# DiffLogger FIXES ITS HEADER ON THE FIRST ROW AND WRITES VALUES IN DICT ORDER
	sample_keys = [f'{k}_{i}' for k in ('ppv','tpr','acc','iou','dice') for i in range(n_classes)] + ['closs']

	total_time_t0 = time.perf_counter()

	for epoch in range(epochs):
		print(f'--- EPOCH {epoch:04} ---')

		# TRAIN
		start_time = time.perf_counter()
		train_loss = train(model,gpu_aug,diffusion,dataloaders['training'],optimizer,scheduler,device,n_classes)
		print(f"train loss (MSE): {train_loss:.5f}")

		# VALIDATE
		valid_loss = validate(model,gpu_norm,diffusion,dataloaders['validation'],device,n_classes)
		print(f"valid loss (MSE): {valid_loss:.5f}")
		train_time = time.perf_counter() - start_time

		print(f"Train+validation time: {train_time:.2f} secs.")

		# SAMPLED SEGMENTATION METRICS (SUBSET) -- USE THESE FOR MODEL SELECTION
		sample_metrics = (epoch+1) % 5 == 0

		if sample_metrics:
			sampling_time_t0 = time.perf_counter()
			ce_result = validate_with_sampling(model,gpu_norm,diffusion,sample_dloader,device,n_classes=n_classes)
			sampling_time_t1 = time.perf_counter()
			assert list(ce_result) == sample_keys, f"sampled columns {list(ce_result)} != {sample_keys}"
		else:
			ce_result = {k:float('nan') for k in sample_keys}

		# COLLECT METRICS & RESULTS
		results = {'epoch':epoch,'tloss':train_loss,'vloss':valid_loss}
		results.update(ce_result)

		if sample_metrics:
			print(f"CE: {results['closs']:.5f} | IoU: {results['iou_1']:.5f} | Dice: {results['dice_1']:.5f}")
			print(f"Sampling time: {sampling_time_t1-sampling_time_t0:.2f} secs.")

		# LOG WHOLE EPOCH
		logger.log(results)

		# SAVE CHECKPOINTS -- SKIP WARMUP; HIGHEST SAMPLED iou_1 SELECTS THE BEST
		if sample_metrics and (epoch+1) > 10 and results['iou_1'] > best_iou:
			best_iou = results['iou_1']
			best_iou_epoch = epoch
			pth_path = utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,tag='full')
			tracker.update(pth_path)

	# STDOUT LOG OVERALL
	total_time_t1 = time.perf_counter()
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	print(f'Total time: {(total_time_t1-total_time_t0)/60:.1f} min')
	print(f'Best epoch: {best_iou_epoch} | IoU: {best_iou:.5f}')
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')


################################################################################
# MAIN/SETUP
################################################################################
if __name__ == '__main__':

	args = parse_args()
	hp   = utils.load_hyperparameters(args)
	utils.set_seed(hp['seed'])
	device = torch.device(f'cuda:{args.device}')


	# CHECK HYPERPARAMETERS
	lrate,decay,epochs,batch = hp['lrate'],hp['decay'],hp['epochs'],hp['batch']
	assert hp['arch'] in ['cnn','vit'], f"Unknown arch {hp['arch']}"
	assert hp['cond_encoder'] in ['cnn','vit'], f"Unknown cond_encoder {hp['cond_encoder']}"
	assert lrate > 0 and decay >= 0, f"Got lrate {lrate}, decay {decay}"
	assert epochs > 10, f"Got {epochs} epochs; 10 are warmup"
	assert (hp['vit_layers'] >= 1) if hp['arch'] == 'vit' else (hp['vit_layers'] == 0), f"Got vit_layers {hp['vit_layers']} for arch {hp['arch']}"

	# 2x2 DESIGN: CONDITION ENCODER FROM SCRATCH OR BT, DENOISER FROM SCRATCH OR diff_pretrain
	cond_init = hp.get('cond_init','scratch')
	diff_init = hp.get('diff_init','scratch')
	assert cond_init in ['scratch','bt'], f"Unknown cond_init {cond_init}"
	assert diff_init in ['scratch','pretrain'], f"Unknown diff_init {diff_init}"

	# CONDITION ENCODER LRATE = lrate * enc_ratio (1.0 = SAME AS THE DENOISER, 0 = FROZEN)
	enc_ratio = hp.get('enc_ratio',1.0)
	assert enc_ratio >= 0, f"Got enc_ratio {enc_ratio}"
	assert enc_ratio > 0 or cond_init == 'bt', "enc_ratio 0 freezes the condition encoder: only meaningful with cond_init 'bt'"

	# GUIDED DENOISER: PREDICTS THE NOISE ON THE MASK, CONDITIONED ON THE 4 BANDS
	n_classes = hp.get('labels',2)
	seg_net = models.ConditionedDenoiser(
		hp['id'],
		arch=hp['arch'],
		cond_encoder=hp['cond_encoder'],            # TYPE OF THE BT CHECKPOINT IF cond_init == 'bt'
		cnn_layers=hp['cnn_layers'],
		vit_layers=max(hp['vit_layers'],1),         # IGNORED BY cnn
		init_chan=hp['init_chan'],                  # SAME AS THE diff_pretrain ROW IF diff_init == 'pretrain'
		init_chan_cond=hp['init_chan_cond'],        # THE BT ROW'S init_chan IF cond_init == 'bt'
		cond_cnn_layers=hp.get('cond_cnn_layers',2),
		mlp_ratio=hp.get('mlp_ratio',4),
		time_dim=hp.get('time_dim',256),
		in_channels=4,
		mask_channels=mask_channels(n_classes)
	)
	print(f"{hp['arch']} denoiser | {hp['cond_encoder']} condition encoder | cond_init {cond_init} | diff_init {diff_init}")
	print(f'Parameters: {models.get_model_parameter_size(seg_net):,}')

	# PRE-TRAINED WEIGHTS (COPIED INTO --net-dir BY k8s/full_train.sh ONLY WHEN THE ROW ASKS FOR THEM)
	if cond_init == 'bt':
		load_cond_encoder(seg_net,hp)
	if diff_init == 'pretrain':
		load_pretrained_denoiser(seg_net,hp)
	seg_net = seg_net.to(device)


	# FROZEN CONDITION ENCODER: NO GRADIENTS, NOT IN THE OPTIMIZER
	if enc_ratio == 0:
		seg_net.cond_encoder.requires_grad_(False)

	# PER MODULE LRATE: CONDITION ENCODER AT lrate*enc_ratio, EVERYTHING ELSE (DENOISER) AT lrate.
	# NO DECAY ON NORM WEIGHTS AND BIASES (ndim < 2)
	cond_params  = list(seg_net.cond_encoder.parameters()) if enc_ratio > 0 else []
	cond_ids     = {id(p) for p in seg_net.cond_encoder.parameters()}
	denoiser_params = [p for p in seg_net.parameters() if id(p) not in cond_ids]
	param_groups = []
	for params,lr in ((denoiser_params,lrate),(cond_params,lrate*enc_ratio)):
		if params:
			param_groups.append({'params':[p for p in params if p.ndim >= 2],'lr':lr,'weight_decay':decay})
			param_groups.append({'params':[p for p in params if p.ndim < 2],'lr':lr,'weight_decay':0.0})
	optimizer = torch.optim.AdamW(param_groups,lr=lrate,fused=device.type == 'cuda') # single fused kernel for the whole update?

	n_train = models.get_model_parameter_size(seg_net,trainable_only=True)
	n_total = models.get_model_parameter_size(seg_net)
	print(f'Trainable parameters: {n_train:,} / {n_total:,} | condition encoder lrate {lrate*enc_ratio:g} (enc_ratio {enc_ratio})')


	# SCHEDULER -- 10-EPOCH LINEAR WARMUP + COSINE ANNEALING W/O RESTARTS, FLOOR AT lrate/cosine_steps
	warmup_steps = 10
	cosine_steps = epochs-warmup_steps
	min_factor = 1.0 / cosine_steps
	cosine_factor = lambda e: min_factor + (1.0-min_factor) * 0.5 * (1.0 + math.cos(math.pi * min(e,cosine_steps) / cosine_steps))
	warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-2,end_factor=1.0,total_iters=warmup_steps)
	cosine_sched = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=cosine_factor)
	scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_sched,cosine_sched],milestones=[warmup_steps])

	# COMPILE AFTER BUILDING THE OPTIMIZER; save_checkpoint UNWRAPS _orig_mod
	seg_net = torch.compile(seg_net,mode='reduce-overhead') # CUDA graphs: fewer kernel launches per forward

	# LOAD DATASET TO RAM
	bands_tr,masks_tr = dataloader.load_cached_split(args.data_dir,split='training')
	bands_va,masks_va = dataloader.load_cached_split(args.data_dir,split='validation')
	dset_tr = dataloader.SentinelFastDataset(bands_tr,masks_tr)
	dset_va = dataloader.SentinelFastDataset(bands_va,masks_va)

	gpu_aug  = dataloader.GPULabeledTransform().to(device) #supervised augmentations
	gpu_norm = dataloader.GPUValidationTransform().to(device)

	# DATALOADERS
	dloader_tr = DataLoader(
		dataset=dset_tr,
		shuffle=True,
		drop_last=True,
		batch_size=batch,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6,
		persistent_workers=True
	)

	dloader_va = DataLoader(
		dataset=dset_va,
		shuffle=False,
		drop_last=False,
		batch_size=batch,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6,
		persistent_workers=True
	)
	dataloaders = {'training':dloader_tr,'validation':dloader_va}

	# FIXED SUBSET FOR DDIM-SAMPLED METRICS: 512 CHIPS EVENLY SPACED OVER THE (SORTED) VALIDATION SET
	sample_idx = np.unique(np.linspace(0,len(dset_va)-1,min(512,len(dset_va)),dtype=int))
	sample_dloader = DataLoader(
		Subset(dset_va,sample_idx.tolist()),
		batch_size=2*hp['batch'], # no gradients -> room for larger batches, fewer sequential DDIM calls
		drop_last=False,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6,
		persistent_workers=True
	)

	# FORWARD + REVERSE DIFFUSION
	diffusion = GaussianDiffusion(seg_net,timesteps=1000,ddim_steps=hp.get('ddim_steps',100),device=device) # GUIDED; x0 (MASK) CLIPPED TO [-1,1]

	train_and_validate(seg_net,gpu_aug,gpu_norm,dataloaders,sample_dloader,optimizer,scheduler,diffusion,device,n_classes=n_classes,epochs=epochs)
