'''
Pretrain an unguided denoiser (models.UnconditionedDenoiser) on the 4 VNIR bands.
The model learns to predict the noise added to normalized bands at a random step t.
Its weights later initialize the diffusion segmentation model (diff_train).

UNGUIDED: GaussianDiffusion(...,guided=False), LOSS diffusion.p_loss(x0,t).
'''

import os
import math
import torch
from torch.utils.data import DataLoader
import time
import argparse

import dataloader
import utils
import models
from diffusion import GaussianDiffusion

LOG_DIR   = None
MODEL_DIR = None

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
	optional.add_argument('--device',required=False,type=int,default=0,help='Override default GPU id 0')

	# LOAD
	args = parser.parse_args()

	# CHECK REQUIRED
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"

	# CHECK OPTIONAL
	assert args.device >= 0, f"Got negative arg for GPU id {args.device}"
	if args.device > 0:
		assert args.device < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."

	# SET GLOBAL VARS -- skip passing as args in train_and_validate()
	global LOG_DIR
	global MODEL_DIR
	LOG_DIR = args.log_dir
	MODEL_DIR = args.net_dir

	# RETURN
	return args


def calculate_fid(dataloader,encoder):
	'''
	Calculate FID in VRAM. TODO
	'''
	pass


def calculate_lpips(dataloader,encoder):
	'''
	Calculate intraset paired LPIPS in VRAM. TODO
	'''
	pass


def validation_set_statistics(dataloader,encoder):
	'''
	Calculate feature mu and sigma of the entire validation set used by FID metric. TODO
	'''
	pass


def train(model,gpu_aug,diffusion,loader,optimizer,scheduler,device):
	'''
	Train unguided model on VNIR bands. Noise-prediction (MSE) loss.
	'''
	model.train()

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	for vnir,mask in loader:

		# TO GPU, AUGMENT & NORMALIZE -- THE MASK IS NOT USED
		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)
		x0,_ = gpu_aug(vnir,mask) # NORMALIZED BANDS ARE x0

		# RANDOM TIMESTEPS
		t = torch.randint(0,diffusion.timesteps,(x0.shape[0],),device=device,dtype=torch.long)

		# FORWARD -- NOISE x0 TO x_t, PREDICT THE NOISE
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			loss = diffusion.p_loss(x0,t)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
		optimizer.step()

		# LOSS IS A BATCH MEAN -- WEIGHT BY BATCH SIZE FOR A PER-SAMPLE MEAN
		loss_sum  += loss.detach() * x0.size(0)
		n_samples += x0.size(0)

	scheduler.step()

	# SYNC
	return {'tloss':(loss_sum/n_samples).item()}


@torch.no_grad()
def validate(model,gpu_norm,diffusion,loader,device):
	'''
	Noise-prediction (MSE) loss over the full validation set. t and noise are drawn
	from a fixed seed, so every epoch is scored on the same (x0,t,noise) triples.
	'''
	model.eval()

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	# FORK THE RNG SO THE FIXED SEED DOES NOT RESET THE TRAINING RNG STREAM
	with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
		torch.manual_seed(VAL_SEED)

		for vnir,mask in loader:

			vnir = vnir.to(device,non_blocking=True)
			mask = mask.to(device,non_blocking=True)
			x0,_ = gpu_norm(vnir,mask)

			t = torch.randint(0,diffusion.timesteps,(x0.shape[0],),device=device,dtype=torch.long)

			with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
				loss = diffusion.p_loss(x0,t)

			loss_sum  += loss * x0.size(0)
			n_samples += x0.size(0)

	# SYNC
	return {'vloss':(loss_sum/n_samples).item()}


@torch.no_grad()
def sample_and_score():
	'''
	Calculate FID LPIPS metrics over a set sampled/generated images vs validation features
	'''
	# SAMPLE ALL PREDEFINED NUMBER OF IMAGES

	# CALCULATE FID
	pass

	# CALCULATE INTRA LPIPS
	pass


def train_and_validate(model,gpu_aug,gpu_norm,diffusion,dataloaders,optimizer,scheduler,device,epochs=100):
	'''
	Train, validate (MSE), log, and keep the best checkpoint by validation MSE.
	'''
	logger  = utils.DiffLogger(f'{LOG_DIR}/diff_{model.model_id:03}.tsv')
	tracker = utils.RecentBestTracker(n=1)
	best_loss  = float('inf')
	best_epoch = 0

	# 0.PRECALCULATE VALIDATION FEATURE STATISTICS -- TODO (FID)
	validation_set_statistics()

	total_time_t0 = time.perf_counter()

	for epoch in range(epochs):

		print(f'--- EPOCH {epoch:04} ---')
		epoch_time_t0 = time.perf_counter()

		# 1.TRAIN
		tr_result = train(model,gpu_aug,diffusion,dataloaders['training'],optimizer,scheduler,device)

		# 2.VALIDATE (MSE ONLY)
		va_result = validate(model,gpu_norm,diffusion,dataloaders['validation'],device)

		# 3.SCORE FID,LPIPS -- TODO, NEEDS UNGUIDED SAMPLING IN diffusion.py
		sample_flag = (epoch+1) % 10 == 0

		if sample_flag:
			scores = sample_and_score()
		else:
			# dummy in logs
			scores = {}

		epoch_time_t1 = time.perf_counter()

		# LOG
		logger.log({'epoch':epoch,**tr_result,**va_result,**scores})
		print(f"[T] LOSS: {tr_result['tloss']:.5f} | [V] LOSS: {va_result['vloss']:.5f}")

		# CHECKPOINTS -- SKIP WARMUP; LOWEST VALIDATION MSE SELECTS THE BEST
		if (epoch+1)>10 and va_result['vloss'] < best_loss:
			best_loss  = va_result['vloss']
			best_epoch = epoch
			pth_path = utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,tag='diff')
			tracker.update(pth_path)

		# STDOUT LOG EPOCH TIME
		print(f"Epoch time: {epoch_time_t1-epoch_time_t0:.2f} secs.")

	total_time_t1 = time.perf_counter()
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

	# STDOUT LOG OVERALL
	print(f'Total time: {(total_time_t1-total_time_t0)/60:.1f} min')
	print(f'Best epoch: {best_epoch} | MSE: {best_loss:.5f}')
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')


@torch.no_grad()
def plot_samples(model,diffusion,device):
	'''
	Plot some images generated from random noise. TODO -- NEEDS UNGUIDED SAMPLING IN diffusion.py
	'''
	pass


################################################################################
# MAIN/SETUP
################################################################################
if __name__ == '__main__':

	# PARSE ARGS/FILES, SET SEED
	args = parse_args()
	hp   = utils.load_hyperparameters(args)
	utils.set_seed(hp['seed'])
	device = torch.device(f'cuda:{args.device}')

	# CHECK HYPERPARAMETERS
	lrate,decay,epochs,batch = hp['lrate'],hp['decay'],hp['epochs'],hp['batch']
	assert hp['arch'] in ['cnn','vit'], f"Unknown arch {hp['arch']}"
	assert lrate > 0 and decay >= 0, f"Got lrate {lrate}, decay {decay}"
	assert epochs > 10, f"Got {epochs} epochs; 10 are warmup"
	assert (hp['vit_layers'] >= 1) if hp['arch'] == 'vit' else (hp['vit_layers'] == 0), f"Got vit_layers {hp['vit_layers']} for arch {hp['arch']}"
	timesteps = hp.get('timesteps',1000)

	# THE UNGUIDED DENOISER -- PREDICTS THE NOISE ON ALL 4 BANDS
	ddpm = models.UnconditionedDenoiser(
		hp['id'],
		arch=hp['arch'],
		cnn_layers=hp['cnn_layers'],
		vit_layers=max(hp['vit_layers'],1), # IGNORED BY cnn
		init_chan=hp['init_chan'],
		mlp_ratio=hp.get('mlp_ratio',4),
		time_dim=hp.get('time_dim',256),
		in_channels=4,
		out_channels=4
	)
	ddpm.to(device)
	print(f"{hp['arch']} | init_chan {hp['init_chan']} | cnn_layers {hp['cnn_layers']} | vit_layers {hp['vit_layers']} | batch {batch} | lrate {lrate} | decay {decay}")
	print(f'Parameters: {models.get_model_parameter_size(ddpm):,}')

	# NO DECAY ON NORM WEIGHTS AND BIASES (ndim < 2)
	params = list(ddpm.parameters())
	param_groups = [
		{'params':[p for p in params if p.ndim >= 2],'weight_decay':decay},
		{'params':[p for p in params if p.ndim < 2],'weight_decay':0.0}
	]
	optimizer = torch.optim.AdamW(param_groups,lr=lrate,fused=device.type == 'cuda') # single fused kernel for the whole update

	# SCHEDULER -- 10-EPOCH LINEAR WARMUP + COSINE ANNEALING W/O RESTARTS, FLOOR AT lrate/cosine_steps
	warmup_steps = 10
	cosine_steps = epochs-warmup_steps
	min_factor = 1.0 / cosine_steps
	cosine_factor = lambda e: min_factor + (1.0-min_factor) * 0.5 * (1.0 + math.cos(math.pi * min(e,cosine_steps) / cosine_steps))
	warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-2,end_factor=1.0,total_iters=warmup_steps)
	cosine_sched = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=cosine_factor)
	scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_sched,cosine_sched],milestones=[warmup_steps])

	# COMPILE AFTER BUILDING THE OPTIMIZER; save_checkpoint UNWRAPS _orig_mod
	ddpm = torch.compile(ddpm)

	# LOAD DATASET TO RAM
	bands_tr,masks_tr = dataloader.load_cached_split(args.data_dir,split='training')
	bands_va,masks_va = dataloader.load_cached_split(args.data_dir,split='validation')

	# GEOMETRIC AUGS ONLY: GAIN WOULD SHIFT THE BAND DISTRIBUTION THE MODEL IS LEARNING
	gpu_aug  = dataloader.GPULabeledTransform(gain=0.0).to(device)
	gpu_norm = dataloader.GPUValidationTransform().to(device)

	dloader_tr = DataLoader(
		dataset=dataloader.SentinelFastDataset(bands_tr,masks_tr),
		shuffle=True,
		drop_last=True,
		batch_size=batch,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6,
		persistent_workers=True
	)

	dloader_va = DataLoader(
		dataset=dataloader.SentinelFastDataset(bands_va,masks_va),
		shuffle=False,
		drop_last=False,
		batch_size=batch,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6,
		persistent_workers=True
	)
	dataloaders = {'training':dloader_tr,'validation':dloader_va}

	# DIFFUSION PROCESS AROUND THE (COMPILED) DENOISER
	# SAMPLED x0 CLIPPED PER BAND TO THE NORMALIZED uint8 RANGE
	x0_min = ((0.0-dataloader.MEAN)/dataloader.STDD).flatten()
	x0_max = ((255.0-dataloader.MEAN)/dataloader.STDD).flatten()
	diffusion = GaussianDiffusion(ddpm,timesteps=timesteps,ddim_steps=hp.get('ddim_steps',100),device=device,guided=False,x0_min=x0_min,x0_max=x0_max)

	train_and_validate(ddpm,gpu_aug,gpu_norm,diffusion,dataloaders,optimizer,scheduler,device,epochs=epochs)
