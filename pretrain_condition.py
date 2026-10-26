import os
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
from models import BarlowTwins

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
	required.add_argument('-p','--params',required=True,help='JSON hyperparameter file.')
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

	# SET GLOBAL VARS
	global LOG_DIR
	global MODEL_DIR
	LOG_DIR = args.log_dir
	MODEL_DIR = args.net_dir

	# RETURN
	return args

@torch.no_grad
def embedding_std(emb):
	'''
	Mean per-dimension std of l2-normalized embeddings, scaled by sqrt(D).
	~1 for well-spread embeddings, ->0 under collapse.
	'''
	# emb is [B,D]
	emb = torch.nn.functional.normalize(emb.float(),dim=1) #unit-length vectors
	return (emb.std(dim=0).mean() * emb.shape[1]**0.5) # no SYNC need .item() downstream

################################################################################
# TRAIN BT
################################################################################
def train(model,gpu_aug,loss_fn,dataloader,optimizer,scheduler,device):

	sums        = torch.zeros(4,device=device)
	batch_count = torch.zeros(1,device=device)

	model.train()
	for X in dataloader:

		# PASS BATCH, AUGMENT, & NORMALIZE
		X = X.to(device,non_blocking=True)
		X1,X2 = gpu_aug(X)

		# FORWARD
		with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
			z1,z2 = model(X1,X2)

		# LOSS
		loss,on_diag,off_diag = loss_fn(z1,z2)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		# CHECK COLLAPSE
		std = embedding_std(z1)

		# LOGS
		sums += torch.stack([loss.detach(),on_diag,off_diag,std])
		batch_count += 1

	scheduler.step()

	loss,diag,offd,std = (sums/batch_count).tolist() # >>>>> CPU-GPU SYNC <<<<<
	return {'loss':loss,'diag':diag,'offd':offd,'tstd':std}


def train_epochs(model,gpu_aug,loss_fn,dataloader,optimizer,scheduler,device,epochs=1000):

	logger  = utils.BTLogger(f'{LOG_DIR}/bt_{model.model_id:03}.tsv')
	# tracker = utils.RecentBestTracker(n=3)

	total_time_t0 = time.perf_counter()
	for epoch in range(epochs):

		print(f'--- EPOCH {epoch:04} ---')
		epoch_time_t0 = time.perf_counter()

		t_result = train(model,gpu_aug,loss_fn,dataloader,optimizer,scheduler,device)
		print(f"[T] LOSS: {t_result['loss']:.5f} | ",end='')
		print(f"ON_DIAG:  {t_result['diag']:.5f} | ",end='')
		print(f"OFF_DIAG: {t_result['offd']:.5f} | ",end='')
		print(f"STD:{t_result['tstd']:.5f}")
		
		logger.log(t_result)

		epoch_time_t1 = time.perf_counter()
		print(f"Epoch time: {epoch_time_t1-epoch_time_t0:.2f} secs.")

	path = utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,tag='bt')

	total_time_t1 = time.perf_counter()
	print(f'Total time: {(total_time_t1-total_time_t0)/60:.1f} min.')
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')


################################################################################
# SETUP/MAIN
################################################################################
if __name__ == '__main__':
	args = parse_args()
	hp   = utils.load_hyperparameters(args)
	utils.set_seed(hp['seed'])
	device = torch.device(f'cuda:{args.device}')

	bt_net = BarlowTwins(
		model_id=hp['id'],
		init_chan=hp['init_chan'],
		cnn_layers=hp['cnn_layers'],
		proj_dim=hp['proj_dim'],
		encoder=hp.get('encoder','cnn')
	).to(device)
	bt_net = torch.compile(bt_net)

	loss_fn = models.BarlowTwinsLoss(proj_dim=hp['proj_dim'],lambd=hp['bt_lambda']).to(device)

	# NO DECAY ON NORM WEIGHTS AND BIASES (ndim < 2)
	decay_params    = [p for p in bt_net.parameters() if p.ndim >= 2]
	no_decay_params = [p for p in bt_net.parameters() if p.ndim < 2]
	optimizer = torch.optim.AdamW([
		{'params':decay_params,'weight_decay':hp['decay']},
		{'params':no_decay_params,'weight_decay':0.0}],
		lr=hp['lrate'])

	# SCHEDULER -- COSINE ANNEALING W/O RESTARTS
	warmup_steps = 10
	cosine_steps = hp['epochs']-warmup_steps
	min_rate = hp['lrate'] / cosine_steps
	warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-2,end_factor=1.0,total_iters=warmup_steps)
	cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cosine_steps,eta_min=min_rate)
	scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_sched,cosine_sched],milestones=[warmup_steps])

	bands,_ = dataloader.load_cached_split(args.data_dir,split='training')
	gpu_aug = dataloader.GPUPairedTransform(crop=192).to(device) #BT augmentations
	# gpu_norm = dataloader.GPUValidationTransform().to(device)
	dloader = DataLoader(
		dataset=dataloader.TwinFastDataset(bands),
		shuffle=True,
		drop_last=True,
		batch_size=hp['batch'],
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=8,
		persistent_workers=True
	)

	train_epochs(bt_net,gpu_aug,loss_fn,dloader,optimizer,scheduler,device,epochs=hp['epochs'])

