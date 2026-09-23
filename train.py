import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import argparse
import json
import inspect

from models import S2SegDiff
from diffusion import GaussianDiffusion, mask_to_x0, x0_to_mask
from dataloader import SentinelDataset


class Logger():
	def __init__(self,path,n_classes):
		'''
		path: str
			The file path to the text file where we log.

		head: [str]
			The column names to be included.
		'''
		self.path = path
		self.n_classes = n_classes

		header = ['tloss','vloss']
		per_class = ('tacc','ttpr','tppv','tiou','tdic','vacc','vtpr','vppv','viou','vdic')
		for prefix in per_class:
			header += [f'{prefix}{c}' for c in range(n_classes)]

		self.header = header
		self.per_class = per_class

		with open(self.path,'w') as fp:
			fp.write('\t'.join(header)+'\n')


class RecentBestTracker:
	'''
	Keeps track of 'n' .pth checkpoints saved as best. 
	Updates queue and removes files no longer needed.
	'''

	def __init__(self,n=3):
		self.n = n
		self.paths = [] #FIFO queue for best 3 recent

	def update(self,path):
		self.paths.append(path)
		if len(self.paths) > self.n:
			old_path = self.paths.pop(0)
			if os.path.exists(old_path):
				os.remove(old_path)	

	def epochs(self):
		return ", ".join([p.split('_')[-1][1:3] for p in self.paths])


def save_checkpoint(path,model,optim,epoch,t_loss,v_loss,tag):
	'''
	Saves model+optim+scaler state as .pth.tar 
	'''
	save_path = f'{path}/{tag}_{model.model_id:03}_e{epoch:02}.pth.tar'

	# SAVE UNCOMPILED IF ALREAD COMPILED
	raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

	# SET CHECKPOINT AND WRITE
	checkpoint = {'epoch': epoch,
					't_loss': t_loss,
					'v_loss': v_loss,
					'model_state_dict': raw_model.state_dict(),
					'optim_state_dict': optim.state_dict()}
	torch.save(checkpoint,save_path)

	# RETURN PATH STR
	return save_path


def set_seed(seed,cuda=True):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)  # If using CUDA
		torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False #Am I losing speed here?
	os.environ['PYTHONHASHSEED'] = str(seed)


@torch.no_grad()
def calculate_metrics(confmat):
	'''
	Calculate precision, recall, accuracy, and IoU for a confusion matrix tensor.
	'''

	# Add stuff
	TP = confmat.diagonal()
	FP = confmat.sum(dim=0) - TP
	FN = confmat.sum(dim=1) - TP
	TN = confmat.sum() - TP - FP - FN
	# eps = 0.0000000001 #<--- uses .clamp() instead

	# the metrics
	ppv = TP / (TP + FP).clamp(min=1) #precision
	tpr = TP / (TP + FN).clamp(min=1) #recall
	acc = (TP+TN) / (TP+FN+FP+TN).clamp(min=1) #accuracy
	iou = TP / (TP + FN + FP).clamp(min=1) #Intersection-over-Union
	dice = 2*TP/(2*TP+FP+FN).clamp(min=1) #Dice score

	return ppv,tpr,acc,iou,dice


@torch.no_grad()
def update_confusion_matrix(confmat,T,Y,n_classes):
	'''
	Update a confusion matrix tensor in gpu. Per-pixel classification.
	'''
	# Vectorized to be [0,1,2,3] == [TN,FP,FN,TP]
	idx = (T.flatten()*n_classes + Y.flatten()).to(torch.int64)
	binc = torch.bincount(idx,minlength=n_classes*n_classes)
	confmat += binc.view(n_classes,n_classes)


def calculate_cross_entropy(x0_pred, true_mask, eps=1e-6):
	'''
	Pixel-wise cross-entropy of the sampled x0 against the true mask.
	x0_pred: [B,num_classes,H,W] in [-1,1] (soft one-hot, see mask_to_x0); 
	true_mask: [B,H,W] integer class map.
	x0_pred is mapped back to [0,1]; its log is passed as logits, so cross_entropy's softmax
	renormalizes it over classes into per-pixel probabilities.
	'''
	probs = ((x0_pred + 1) / 2).clamp(min=eps)
	return F.cross_entropy(torch.log(probs), true_mask.long()).item()


def format_stdout_metrics(prefix, loss, iou, dice):
	s = f'[{prefix}] LOSS: {loss:.5f} '
	s += f' | IoU_0: {iou[0]:.5f} | IoU_1: {iou[1]:.5f}'
	s += f' | Dice_0: {dice[0]:.5f} | Dice_1: {dice[1]:.5f}'
	return s


def load_hyperparameters(args,print=False):
	# LOAD FILE
	with open(args.params,'r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	assert len(hp_list) > 0, f"Got empty file for {args.params}"

	# SET IDs AS KEYS and CHECK
	hp_list_indexed = {row['id']:row for row in hp_list}
	assert args.id in hp_list_indexed, f"model id '{args.id}' not in hyperparameter file {args.params}"

	# SET DICT FOR CURRENT MODEL
	HP = hp_list_indexed[args.id]

	# CHECK DICT
	pass

	# PRINT
	if print:
		pass

	return HP



def parse_args():
	'''
	Load args.
	'''
	# DEFINITION
	parser = argparse.ArgumentParser()
	required = parser.add_argument_group('Required arguments')
	required.add_argument('--data-dir',required=True,help='Input dataset directory.')
	required.add_argument('--net-dir',required=True,help='Output dir for trained model weights.')
	required.add_argument('--log-dir',required=True,help='Output dir for training logs.')
	required.add_argument('-p','--params',required=True,help='JSON hyperparameter file.')
	required.add_argument('--id',required=True,type=int,help='model id in hyperparameter file.')

	# OPTIONAL
	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--workers',required=False,type=int,default=4,help='Set num_workers args.')
	optional.add_argument('--gpu',required=False,type=int,default=0,help='Override default GPU 0')

	# LOAD
	args = parser.parse_args ()

	# CHECK REQUIRED
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"

	# CHECK OPTIONAL
	assert args.gpu >= 0, f"Got negative arg for GPU id {args.gpu}"
	if args.gpu > 0:
		assert args.gpu < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."	
	assert args.workers in range(0,8), "Got out of range arg for nr of workers"

	# RETURN
	return args


def train(model,diffusion,loader,optimizer,device,n_classes=2):

	loss_sum   = torch.zeros(1,device=device)
	model.train()

	for rgb,lbl in loader:

		rgb.to(device)
		lbl.to(device)

		x_0 = mask_to_x0(lbl,n_classes)
		t   = torch.randint(0,diffusion.timesteps,(rgb.shape[0],),device=device,dtype=torch.long)

		with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
			output = 
			loss   = 

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loss_sum += loss.detach() * rgb.size(0)

	return loss_sum

@torch.no_grad()
def validate(sample=False):
	model.eval()
	loss_sum = torch.zeros(1,device=device)
	gpu_cmat = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)

	for rgb,lbl in loader:

		rgb.to(device)
		lbl.to(device)

		x0 = mask_to_x0(lbl,n_classes)
		t  = torch.randint(0,diffusion_timesteps, rgb)

		with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True)
			loss = p_loss(x0,rgb,t)
		loss_sum += loss.detach() * rgb.size(0)

		if sample:
			x0_pred   = diffusion.p_sample_loop(rgb,mask_channels=n_classes)
			pred_mask = x0_to_mask(x0_pred)
			# total_ce  = 
			# preds = torch.argmax(logits, dim=1)
			update_confusion_matrix(gpu_cmat,lbl,pred_mask,n_classes)

	return loss_sum,gpu_cmat


def train_and_validate(args):

	device = 

	hp = load_hyperparameters(args)
	b_size = 
	model  = 
	optimizer = torch.optim.AdamW(model.paramters(),lr=hp['lrate'])

	train_dataset = SentinelDataset(f"{args.data_dir}/training",n_bands,n_labels=2,transform=None)
	valid_dataset = SentinelDataset(f"{args.data_dir}/validation",n_bands,n_labels=2,transform=None)
	train_dloader = DataLoader(
		train_dataset,
		batch_size=b_size,
		drop_last=False,
		shuffle=True,
		num_workers=args.num_workers)
	valid_dloader = DataLoader(
		valid_dataset,
		batch_size=b_size,
		drop_last=False,
		shuffle=False,
		num_workers=args.num_workers)

	diffusion = GaussianDiffusion(model,timesteps,device)

	log_file_path = f'{args.log_dir}/epochs_{model.model_id:03}.tsv'
	logger        = Logger(log_file_path,n_classes=2)	

	for epoch in range(epochs):

		# TRAIN
		train_loss_sum,train_cmat = train(model,diffusion,train_dloader,optimizer,device,n_classes=2)

		# VALIDATE
		sample_metrics = (epochs+1) % 5 == 0
		valid_loss_sum,valid_cmat = validate(model,diffusion,valid_dloader,device,n_classes=2,sample=sample_metrics)

		# LOG EPOCH
		train_loss = train_loss_sum.item()/len(train_dataset)
		valid_loss = valid_loss_sum.item()/len(valid_dataset)
		va_metrics = calculate_metrics(valid_cmat.cpu())
		logger.log(results)


		# SAVE CHECKPOINTS


if __name__ == '__main__':
	pass
