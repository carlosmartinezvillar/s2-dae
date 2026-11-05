import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import argparse
# import inspect

import models
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


def calculate_fid(dataloader,encoder):
	'''
	Calculate FID in VRAM.
	'''
	pass


def calculate_lpips(dataloader,encoder):
	'''
	Calculate intraset paired LPIPS in VRAM.
	'''
	pass


def validation_set_statistics(dataloader,encoder):
	'''
	Calculate feature mu and sigma of the entire validation set used by FID metric.
	'''
	pass


def train(model,gpu_aug,diffusion,loader,optimizer,scheduler,device):
	'''
	Train unguided model on VNIR bands.
	'''

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	model.train()

	for vnir,mask in loader:

		# TO DEV & NORM
		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True) #-- not needed
		vnir,_ = gpu_aug(vnir,mask)

		# GET X0 & RANDOM TIMESTEPS
		# x0 = mask_to_x0(mask,2)

		# ADJUST VNIR

		# RANDOM TIMESTEPS
		t  = torch.randint(0,diffusion.timesteps,(vnir.shape[0],),device=device,dtype=torch.long)

		# GET NOISY VNIR

		# FORWARD
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			loss = diffusion.p_loss(x0,vnir,t) # ADJUST TO UNCONDITIONED

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
		optimizer.step()		

		loss_sum += loss.detach() * vnir.size(0)
		n_samples += vnir.size(0)

	scheduler.step()

	return (loss_sum/samp_sum).item() # >>>> CPU-GPU SYNC <<<<


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

		# x0 = mask_to_x0(mask,n_classes)
		t  = torch.randint(0,diffusion.timesteps,(vnir.shape[0],),device=device,dtype=torch.long)

		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			mse_loss = diffusion.p_loss(x0,vnir,t) # ADJUST TO UNCONDITIONED

		mse_loss_sum += mse_loss.detach() * vnir.size(0)
		n_samples    += vnir.size(0)

	return (mse_loss_sum/n_samples).item()


def train_and_validate():

	# 0.PRECALCULATE VALIDATION FEATURE STATISTICS

	# 1.TRAIN

	# 2.VALIDATE (MSE ONLY)

	# 3. SCORE FID,LPIPS

	# LOG

	# CHECKPOINTS
	pass


@torch.no_grad()
def plot_samples(model,diffusion,device):
	'''
	Plot some images generated from random noise.
	'''
	pass


if __name__ == '__main__':

	# SETUP
	args = parse_args()
	hp   = utils.load_hyperparameters(args)
	utils.set_seed(hp['seed'])
	device = torch.device(f'cuda:{args.device}')

	# CHECK HYPERPARAMETERS
	lrate,decay,epochs,batch = hp['lrate'],hp['decay'],hp['epochs'],hp['batch']
	assert hp['encoder'] in ['cnn','vit'], f"Unknown encoder {hp['encoder']}"
	assert hp['decoder'] in ['cnn','vit'], f"Unknown decoder {hp['decoder']}"
	assert lrate > 0 and decay >= 0, f"Got lrate {lrate}, decay {decay}"
	assert epochs > 10, f"Got {epochs} epochs; 10 are warmup"
	has_vit = 'vit' in (hp['encoder'],hp['decoder'])
	assert (hp['vit_layers'] >= 1) if has_vit else (hp['vit_layers'] == 0), f"Got vit_layers {hp['vit_layers']} for {hp['encoder']}-{hp['decoder']}"

	# THE UNGUIDED DDPM MODEL
	ddpm = models.UnconditionedDenoiser()

	# NO DECAY ON NORM WEIGHTS AND BIASES (ndim < 2)
	params = list(ddpm.parameters())
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

	ddpm = torch.compile(ddpm)

	# LOAD DATASET TO RAM
	bands_tr,masks_tr = dataloader.load_cached_split(args.data_dir,split='training')
	bands_va,masks_va = dataloader.load_cached_split(args.data_dir,split='validation')
	dset_tr = dataloader.SentinelFastDataset(bands_tr,masks_tr)
	dset_va = dataloader.SentinelFastDataset(bands_va,masks_va)

	# NORMALIZE? 
	# gpu_norm = dataloader.GPUValidationTransform().to(device)

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
	dloader_sampling = DataLoader(
		Subset(dset_va,sample_idx.tolist()),
		batch_size=2*hp['batch'], # no gradients -> room for larger batches, fewer sequential DDIM calls
		drop_last=False,
		shuffle=False,
		**loader_kwargs)

	diffusion = GaussianDiffusion(ddpm,timesteps=1000,ddim_steps=100,device=device)

	pass