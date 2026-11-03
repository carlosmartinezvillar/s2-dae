'''
Train a basic encoder-decoder U-Net for semantic segmentation.
Encoder-decoder combinations vary between CNN-ViT, ViT-CNN, ViT-ViT, and CNN-CNN.
'''

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import math
import argparse

import dataloader
import utils
import models

LOG_DIR   = None
MODEL_DIR = None

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


def train(model,gpu_aug,dataloader,loss_fn,optimizer,scheduler,device):
	model.train()

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	for vnir,mask in dataloader:

		# TO GPU, AUGMENT & NORMALIZE
		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)
		vnir,mask = gpu_aug(vnir,mask)

		# FORWARD
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			pred = model(vnir)
		loss = loss_fn(pred.float(),mask) # logits [B,2,H,W], class indices [B,H,W]

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
		optimizer.step()

		# LOSS IS A BATCH MEAN -- WEIGHT BY BATCH SIZE FOR A PER-SAMPLE MEAN
		loss_sum  += loss.detach() * mask.size(0)
		n_samples += mask.size(0)

	scheduler.step()

	# SYNC
	tloss = (loss_sum/n_samples).item()
	return {'tloss':tloss}


@torch.no_grad()
def validate(model,gpu_norm,dataloader,loss_fn,device):
	'''
	Validation loss and per-class segmentation metrics.
	'''
	model.eval()

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)
	gpu_cmat  = torch.zeros((2,2),device=device,dtype=torch.int64)

	for vnir,mask in dataloader:

		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)
		vnir,mask = gpu_norm(vnir,mask)

		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			pred = model(vnir)

		loss = loss_fn(pred.float(),mask)
		loss_sum  += loss * mask.size(0)
		n_samples += mask.size(0)
		utils.update_confusion_matrix(gpu_cmat,mask,pred.argmax(dim=1),2)

	# SYNC
	vloss    = (loss_sum/n_samples).item()
	cpu_cmat = gpu_cmat.cpu()

	# RESULT -- FLAT FLOATS FOR THE LOGGER, e.g. iou_0, iou_1
	result = utils.calculate_seg_metrics(cpu_cmat)
	flat_result = {f"{k}_{i}":result[k][i].item() for k in result.keys() for i in range(2)}
	flat_result.update({'vloss': vloss})
	return flat_result


def train_and_validate(model,gpu_aug,gpu_norm,dataloaders,loss_fn,optimizer,scheduler,device,epochs=100):
	logger = utils.SegmentationLogger(f'{LOG_DIR}/base_{model.model_id:03}.tsv')
	tracker = utils.RecentBestTracker(n=1)
	best_iou = 0.0
	best_epoch = 0

	total_time_t0 = time.perf_counter()

	for epoch in range(epochs):

		print(f'--- EPOCH {epoch:04} ---')
		epoch_time_t0 = time.perf_counter()
		tr_result = train(model,gpu_aug,dataloaders['training'],loss_fn,optimizer,scheduler,device)
		va_result = validate(model,gpu_norm,dataloaders['validation'],loss_fn,device)
		epoch_time_t1 = time.perf_counter()

		logger.log({'epoch':epoch,**tr_result,**va_result})
		print(f"[T] LOSS: {tr_result['tloss']:.5f} | [V] LOSS: {va_result['vloss']:.5f} | IoU_1: {va_result['iou_1']:.5f}")

		# SKIP WARMUP; FOREGROUND (CLASS 1) IoU SELECTS THE BEST
		if (epoch+1)>10 and va_result['iou_1'] > best_iou:
			best_iou = va_result['iou_1']
			best_epoch = epoch
			pth_path = utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,tag='base')
			tracker.update(pth_path)

		# STDOUT LOG EPOCH TIME
		print(f"Epoch time: {epoch_time_t1-epoch_time_t0:.2f} secs.")

	total_time_t1 = time.perf_counter()
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

	# STDOUT LOG OVERALL
	print(f'Total time: {(total_time_t1-total_time_t0)/60:.1f} min')
	print(f'Best epoch: {best_epoch} | IoU: {best_iou:.5f}')
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')

	
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
	assert hp['encoder'] in ['cnn','vit'], f"Unknown encoder {hp['encoder']}"
	assert hp['decoder'] in ['cnn','vit'], f"Unknown decoder {hp['decoder']}"
	assert lrate > 0 and decay >= 0, f"Got lrate {lrate}, decay {decay}"
	assert epochs > 10, f"Got {epochs} epochs; 10 are warmup"
	has_vit = 'vit' in (hp['encoder'],hp['decoder'])
	assert (hp['vit_layers'] >= 1) if has_vit else (hp['vit_layers'] == 0), f"Got vit_layers {hp['vit_layers']} for {hp['encoder']}-{hp['decoder']}"

	# SEGMENTATION NETWORK
	seg_net = models.BaseSegmenter(
		hp['id'],
		encoder=hp['encoder'],
		decoder=hp['decoder'],
		init_chan=hp['init_chan'],
		cnn_layers=hp['cnn_layers'],
		vit_layers=max(hp['vit_layers'],1), # IGNORED BY cnn-cnn
		mlp_ratio=hp.get('mlp_ratio',4)
	)
	seg_net.to(device)
	print(f"{hp['encoder']}-{hp['decoder']} | init_chan {hp['init_chan']} | vit_layers {hp['vit_layers']} | batch {batch} | lrate {lrate} | decay {decay}")
	print(f'Parameters: {models.get_model_parameter_size(seg_net):,}')

	# NO DECAY ON NORM WEIGHTS AND BIASES (ndim < 2)
	params = list(seg_net.parameters())
	param_groups = [
		{'params':[p for p in params if p.ndim >= 2],'weight_decay':decay},
		{'params':[p for p in params if p.ndim < 2],'weight_decay':0.0}
	]

	# LOSS AND OPTIMIZER
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(param_groups,lr=lrate)

	# SCHEDULER -- 10-EPOCH LINEAR WARMUP + COSINE ANNEALING W/O RESTARTS, FLOOR AT lrate/cosine_steps
	warmup_steps = 10
	cosine_steps = epochs-warmup_steps
	min_factor = 1.0 / cosine_steps
	cosine_factor = lambda e: min_factor + (1.0-min_factor) * 0.5 * (1.0 + math.cos(math.pi * min(e,cosine_steps) / cosine_steps))
	warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-2,end_factor=1.0,total_iters=warmup_steps)
	cosine_sched = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=cosine_factor)
	scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_sched,cosine_sched],milestones=[warmup_steps])

	# COMPILE AFTER BUILDING THE OPTIMIZER; save_checkpoint UNWRAPS _orig_mod
	seg_net = torch.compile(seg_net)

	# LOAD DATASET TO RAM
	bands_tr,masks_tr = dataloader.load_cached_split(args.data_dir,split='training')
	bands_va,masks_va = dataloader.load_cached_split(args.data_dir,split='validation')

	gpu_aug  = dataloader.GPULabeledTransform().to(device) # Supervised augmentations
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

	train_and_validate(seg_net,gpu_aug,gpu_norm,dataloaders,loss_fn,optimizer,scheduler,device,epochs=epochs)
