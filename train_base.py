import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import argparse

import dataloader
import utils
import models

LOG_DIR   = None
MODEL_DIR = None

# FIXED PROBE SETTINGS -- SAME FOR EVERY BT CHECKPOINT SO PROBE SCORES ARE COMPARABLE
# PROBE_LRATE  = 0.0002
# PROBE_EPOCHS = 100
# PROBE_BATCH  = 32

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
	required.add_argument('--probe-only',action='store_true',help='If set, freeze encoder and fit decoder only.')


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


def train_and_validate(model,gpu_aug,gpu_norm,dataloaders,loss_fn,optimizer,scheduler,device,epochs=100)


if __name__ == '__main__':

	# PARSE ARGS/FILES, SET SEED
	args = parse_args()
	hp   = utils.load_hyperparameters(args)
	utils.set_seed(hp['seed'])
	device = torch.device(f'cuda:{args.device}')


	seg_net = model.BaseSegmenter()
	seg_net.to(device)
	seg_net = torch.compile(seg_net)

	# LOSS AND OPTIMIZER -- ONLY TRAINABLE PARAMS
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(param_groups,lr=hp['lrate'])

	# SCHEDULER -- COSINE ANNEALING W/O RESTARTS
	warmup_steps = 10
	cosine_steps = epochs-warmup_steps
	min_rate = lrate / cosine_steps
	warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-2,end_factor=1.0,total_iters=warmup_steps)
	cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cosine_steps,eta_min=min_rate)
	scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_sched,cosine_sched],milestones=[warmup_steps])

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
		prefetch_factor=8,
		persistent_workers=True
	)

	dloader_va = DataLoader(
		dataset=dataloader.SentinelFastDataset(bands_va,masks_va),
		shuffle=False,
		drop_last=False,
		batch_size=batch,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=8,
		persistent_workers=True
	)
	dataloaders = {'training':dloader_tr,'validation':dloader_va}