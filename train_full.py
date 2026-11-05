import os
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
from models
from diffusion import GaussianDiffusion, mask_to_x0, x0_to_mask, mask_channels


def parse_args():
	'''
	Load args.
	'''
	# DEFINITION
	parser = argparse.ArgumentParser()
	required = parser.add_argument_group('Required arguments')
	required.add_argument('--data-dir',required=True,help='Cache directory with training/ and validation/ built by fast_dataloader.py.')
	required.add_argument('--net-dir',required=True,help='Output dir for trained model weights.')
	required.add_argument('--log-dir',required=True,help='Output dir for training logs.')
	required.add_argument('--params',required=True,help='JSON hyperparameter file.')
	required.add_argument('--id',required=True,type=int,help='model id in hyperparameter file.')

	# OPTIONAL
	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--workers',required=False,type=int,default=4,help='Set num_workers args.')
	optional.add_argument('--device',required=False,type=int,default=0,help='Override default GPU 0')

	# LOAD
	args = parser.parse_args ()

	# CHECK REQUIRED
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"

	# CHECK OPTIONAL
	assert args.device >= 0, f"Got negative arg for GPU id {args.gpu}"
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


def load_encoder_checkpoint(hp,epoch=None):
	pass


def load_denoiser_checkpoint(hp,epoch=None):
	pass


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


def train(model,gpu_aug,diffusion,loader,optimizer,scheduler,device):

	model.train()

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	for vnir,mask in loader:

		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)
		vnir,mask = gpu_aug(vnir,mask)

		x0  = mask_to_x0(mask,2)
		t   = torch.randint(0,diffusion.timesteps,(vnir.shape[0],),device=device,dtype=torch.long)

		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			loss = diffusion.p_loss(x0,vnir,t)

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
		optimizer.step()


		loss_sum  += loss.detach() * vnir.size(0)
		n_samples += vnir.size(0)

	scheduler.step()

	return (loss_sum/n_samples).item() # >>>> CPU-GPU SYNC <<<<


@torch.no_grad()
def validate(model,gpu_norm,diffusion,loader,device):
	'''
	Noise-prediction (MSE) loss over the full validation set.
	'''

	model.eval()

	mse_loss_sum = torch.zeros(1,device=device)
	n_samples    = torch.zeros(1,device=device)

	for vnir,mask in loader:

		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)
		vnir,mask = gpu_norm(vnir,mask)

		x0 = mask_to_x0(mask,n_classes)
		t  = torch.randint(0,diffusion.timesteps,(vnir.shape[0],),device=device,dtype=torch.long)

		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			mse_loss = diffusion.p_loss(x0,vnir,t)

		mse_loss_sum += mse_loss.detach() * vnir.size(0)
		n_samples    += vnir.size(0)

	return (mse_loss_sum/n_samples).item()


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
			
			mask = mask.to(device,non_blocking=True)

			with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
				# x0_pred = diffusion.p_sample_loop(vnir,mask_channels=mask_channels(n_classes))
				x0_pred = diffusion.ddim_sample_loop(vnir,mask_channels=mask_channels(n_classes))
			pred_mask = x0_to_mask(x0_pred)

			ce_loss_sum += calculate_cross_entropy(x0_pred.float(),lbl) * vnir.size(0)
			n_samples   += vnir.size(0)

			utils.update_confusion_matrix(gpu_cmat,mask,pred_mask,n_classes)

	# CPU-GPU SYNC
	closs    = (ce_loss_sum/n_samples).item() 
	cpu_cmat = gpu_cmat.cpu()

	result = utils.calculate_seg_metrics(cpu_cmat)
	flat_result = {f"{k}_{i}":result[k][i].item() for k in result.keys() for i in range(2)}
	flat_result.update({'closs': closs})

	return


def train_and_validate(model,gpu_aug,gpu_norm,dataloaders,optimizer,scheduler,diffusion,device,epochs=100):
 
	# LOG PATHS/BUFFERS
	logger  = utils.DiffLogger(f'{LOG_DIR}/base_{model.model_id:03}.tsv')
	tracker = utils.RecentBestTracker(n=3) 
	best_iou = 0.0
	best_iou_epoch = 0
	# best_dice = 0.0
	# best_dice_epoch = 0

	total_time_t0 = time.perf_counter()

	for epoch in range(epochs):
		print(f'--- EPOCH {epoch:04} ---')

		# TRAIN
		start_time = time.perf_counter()
		train_loss = train(model,gpu_aug,diffusion,dataloaders['training'],optimizer,scheduler,device)
		print(f"train loss (MSE): {train_loss:.5f}")

		# VALIDATE
		valid_loss = validate(model,gpu_norm,diffusion,dataloaders['validation'],device)
		print(f"valid loss (MSE): {valid_loss:.5f}")
		train_time = time.perf_counter() - start_time

		print(f"Train+validation time: {train_time:.2f} secs.")

		# SAMPLED SEGMENTATION METRICS (SUBSET) -- USE THESE FOR MODEL SELECTION
		sample_metrics = (epoch+1) % 5 == 0

		if sample_metrics:
			sampling_time_t0 = time.perf_counter()
			ce_result = validate_with_sampling(model,gpu_norm,diffusion,sample_dloader,device,normalize,n_classes=2)
			sampling_time_t1 = time.perf_counter()
		else:
			ce_loss   = float('nan')
			ce_result = {k:torch.full((2,),float('nan')) for k in ('ppv','tpr','acc','iou','dice')}

		# COLLECT METRICS & RESULTS
		results = {'tloss':train_loss,'vloss':valid_loss,'closs':ce_loss}
		results.update(ce_result)

		if sample_metrics:
			print(f"CE: {results['closs']:.5f} | IoU: {results['iou_1']:.5f} | Dice: {results['dice_1']:.5f}")
			print(f"Sampling time: {sampling_time_t1-sampling_time_t0:.2f} secs.")

		# LOG WHOLE EPOCH
		logger.log(results)

		# SAVE CHECKPOINTS
		if sample_metrics:
			if best_iou < results['iou_1']:
				best_iou = results['iou_1']
				best_iou_epoch = epoch
				pth_path = save_checkpoint(MODEL_DIR,model,optimizer,epoch,tag='full')
				recent_best.update(pth_path)

	# STDOUT LOG OVERALL
	total_time_t1 = time.perf_counter()
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	print(f'Total time: {(total_time_t1-total_time_t0)/60:.1f} min')
	print(f'Best epoch: {best_epoch} | IoU: {best_iou:.5f}')
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')

if __name__ == '__main__':

	args = parse_args()
	hp   = utils.load_hyperparameters(args)
	utils.set_seed(hp['seed'])
	# if hp.get('seed',0) != 0:
		# set_seed(hp['seed'])
	device = torch.device(f'cuda:{args.device}')


	# CHECK HYPERPARAMETERS
	lrate,decay,epochs,batch = hp['lrate'],hp['decay'],hp['epochs'],hp['batch']
	assert hp['encoder'] in ['cnn','vit'], f"Unknown encoder {hp['encoder']}"
	assert hp['decoder'] in ['cnn','vit'], f"Unknown decoder {hp['decoder']}"
	assert lrate > 0 and decay >= 0, f"Got lrate {lrate}, decay {decay}"
	assert epochs > 10, f"Got {epochs} epochs; 10 are warmup"
	has_vit = 'vit' in (hp['encoder'],hp['decoder'])
	assert (hp['vit_layers'] >= 1) if has_vit else (hp['vit_layers'] == 0), f"Got vit_layers {hp['vit_layers']} for {hp['encoder']}-{hp['decoder']}"

	n_classes = hp['labels']
	seg_net = models.ConditionedDenoiser(
		model_id=hp['id'],
		mask_channels=mask_channels(n_classes),
		in_channels=hp['bands'],
		cnn_layers=hp['cnn_layers'],
		vit_layers=hp['vit_layers'],
		channels=hp['channels'],
		mlp_ratio=hp['mlp_ratio']
	)
	seg_net = seg_net.to(device)


	# NO DECAY ON NORM WEIGHTS AND BIASES (ndim < 2)
	params = list(seg_net.parameters())
	param_groups = [
		{'params':[p for p in params if p.ndim >= 2],'weight_decay':decay},
		{'params':[p for p in params if p.ndim < 2],'weight_decay':0.0}
	]
	optimizer = torch.optim.AdamW(param_groups,lr=lrate,fused=device.type == 'cuda') # single fused kernel for the whole update?


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
	# noise     = None # GPUGaussianNoise() to augment (use together with FastTrainTransform)

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
		**loader_kwargs)

	# FORWARD + REVERSE DIFFUSION
	diffusion = GaussianDiffusion(seg_net,timesteps=1000,ddim_steps=100,device=device)

	train_and_validate(model,gpu_aug,gpu_norm,dataloaders,optimizer,scheduler,diffusion,device,epochs=100)
