import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import time
import argparse
import json
import inspect

from model import S2SegDiff
from diffusion import GaussianDiffusion, mask_to_x0, x0_to_mask
from fast_dataloader import FastSentinelDataset, FastTrainTransform, GPUGaussianNoise, GPUNormalize


class Logger():
	def __init__(self,path,n_classes=2):
		'''
		path: str
			The file path to the text file where we log.

		head: [str]
			The column names to be included.
		'''
		self.path = path
		self.n_classes = n_classes

		header = ['tloss','vloss','closs']
		per_class = ('acc','tpr','ppv','iou','dice')
		for prefix in per_class:
			header += [f'{prefix}_{c}' for c in range(n_classes)]

		self.header = header
		self.per_class = per_class

		with open(self.path,'w') as fp:
			fp.write('\t'.join(header)+'\n')

	def log(self,metrics):
		'''
		metrics: Dict
		'''
		# line = '\t'.join([f'{_:.5f}' for _ in stats])

		row = [f"{metrics['tloss']:.5f}",f"{metrics['vloss']:.5f}",f"{metrics['closs']:.5f}"]
		for prefix in self.per_class:
			row += [f'{metrics[prefix][c]:.5f}' for c in range(self.n_classes)]

		with open(self.path,'a') as fp:
			fp.write('\t'.join(row) + '\n')


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


def save_checkpoint(path,model,optim,epoch,tag):
	'''
	Saves model+optim+scaler state as .pth.tar 
	'''
	save_path = f'{path}/{tag}_{model.model_id:03}_e{epoch:02}.pth.tar'

	# SAVE UNCOMPILED IF ALREAD COMPILED
	raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

	# SET CHECKPOINT AND WRITE
	checkpoint = {'epoch': epoch,
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

	return {'ppv':ppv,'tpr':tpr,'acc':acc,'iou':iou,'dice':dice}


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
	Returns a 0-d tensor on x0_pred's device (no .item(), so no GPU-CPU sync).
	'''
	probs = ((x0_pred + 1) / 2).clamp(min=eps)
	return F.cross_entropy(torch.log(probs), true_mask.long())


def load_hyperparameters(args,print=False):
	# LOAD FILE
	with open(args.params,'r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	assert len(hp_list) > 0, f"Got empty file for {args.params}"

	# SET IDs AS KEYS and CHECK
	hp_list_indexed = {row['id']:row for row in hp_list}
	assert args.id in hp_list_indexed, f"model id '{args.id}' not in hyperparameter file {args.params}"

	# SET DICT FOR CURRENT MODEL
	hp_final = hp_list_indexed[args.id]

	# CHECK DICT -- missing
	pass

	# PRINT
	if print:
		pass

	return hp_final


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
	assert args.workers in range(0,9), "Got out of range arg for nr of workers"

	# RETURN
	return args


def train(model,diffusion,loader,optimizer,scheduler,device,normalize,noise=None,n_classes=2):

	loss_sum   = torch.zeros(1,device=device)
	model.train()

	for rgb,lbl in loader:

		rgb = rgb.to(device,non_blocking=True)
		lbl = lbl.to(device,non_blocking=True)
		# if noise is not None:
			# rgb = noise(rgb) # training augmentation (noise part), on the 0-255 scale
		rgb = normalize(rgb)

		x0  = mask_to_x0(lbl,n_classes)
		t   = torch.randint(0,diffusion.timesteps,(rgb.shape[0],),device=device,dtype=torch.long)

		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			loss = diffusion.p_loss(x0,rgb,t)

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
		optimizer.step()
		scheduler.step() # per-batch: linear warmup, then constant

		loss_sum += loss.detach() * rgb.size(0)

	return loss_sum.item()/len(loader.dataset)

@torch.no_grad()
def validate(model,diffusion,loader,device,normalize,n_classes=2):
	'''
	Noise-prediction (MSE) loss over the full validation set.
	'''
	model.eval()
	mse_loss_sum = torch.zeros(1,device=device)

	for rgb,lbl in loader:

		rgb = normalize(rgb.to(device,non_blocking=True))
		lbl = lbl.to(device,non_blocking=True)

		x0 = mask_to_x0(lbl,n_classes)
		t  = torch.randint(0,diffusion.timesteps,(rgb.shape[0],),device=device,dtype=torch.long)

		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			mse_loss = diffusion.p_loss(x0,rgb,t)
		mse_loss_sum += mse_loss.detach() * rgb.size(0)

	return mse_loss_sum.item()/len(loader.dataset)


@torch.no_grad()
def validate_sampling(model,diffusion,loader,device,normalize,n_classes=2,seed=0):
	'''
	Segmentation metrics from DDIM-sampled masks, on a fixed subset of the validation set.
	The sampling noise uses a fixed seed on a forked RNG, so every evaluation sees the
	same noise (comparable across epochs/runs) and the training RNG stream is untouched.
	'''
	model.eval()
	ce_loss_sum = torch.zeros(1,device=device)
	gpu_cmat    = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)
	# use_bf16    = device.type == 'cuda' and torch.cuda.is_bf16_supported()

	with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
		torch.manual_seed(seed)

		for rgb,lbl in loader:

			rgb = normalize(rgb.to(device,non_blocking=True))
			lbl = lbl.to(device,non_blocking=True)

			with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
				# x0_pred = diffusion.p_sample_loop(rgb,mask_channels=n_classes)
				x0_pred = diffusion.ddim_sample_loop(rgb,mask_channels=n_classes)
			pred_mask = x0_to_mask(x0_pred)
			ce_loss_sum += calculate_cross_entropy(x0_pred.float(),lbl) * rgb.size(0)
			update_confusion_matrix(gpu_cmat,lbl,pred_mask,n_classes)

	return ce_loss_sum.item()/len(loader.dataset), gpu_cmat


def train_and_validate(args):

	device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
	hp = load_hyperparameters(args)

	if hp.get('seed',0) != 0:
		set_seed(hp['seed'])

	n_classes = hp['labels']
	model = S2SegDiff(model_id=hp['id'],mask_channels=n_classes,in_channels=hp['bands'],cnn_layers=hp['cnn_layers'],
						vit_layers=hp['vit_layers'],channels=hp['channels'],mlp_ratio=hp['mlp_ratio'])
	model = model.to(device)
	model = torch.compile(model,mode='reduce-overhead') # CUDA graphs: fewer kernel launches per forward

	# WEIGHT DECAY ONLY ON WEIGHT MATRICES/KERNELS -- NOT ON NORMS AND BIASES
	decay_params    = [p for p in model.parameters() if p.ndim >= 2]
	no_decay_params = [p for p in model.parameters() if p.ndim < 2]
	optimizer = torch.optim.AdamW([
		{'params':decay_params,'weight_decay':hp['decay']},
		{'params':no_decay_params,'weight_decay':0.0}],
		lr=hp['lrate'],
		fused=device.type == 'cuda') # single fused kernel for the whole update

	# DATASETS
	# uint8 CHIPS FROM THE .npy CACHE; NORMALIZATION (AND TRAINING NOISE) ON THE GPU
	train_dataset = FastSentinelDataset(f"{args.data_dir}/training",transform=None) # transform=FastTrainTransform() to augment
	valid_dataset = FastSentinelDataset(f"{args.data_dir}/validation",transform=None)
	for ds in (train_dataset,valid_dataset):
		assert ds.n_bands == hp['bands'] and ds.n_labels == n_classes, \
			f"Cache {ds.cache_dir} has bands={ds.n_bands}, labels={ds.n_labels}; hyperparameters need bands={hp['bands']}, labels={n_classes}"

	normalize = GPUNormalize(train_dataset.mean,train_dataset.std,device)
	noise     = None # GPUGaussianNoise() to augment (use together with FastTrainTransform)

	# DATALOADERS
	loader_kwargs = {
		'num_workers':args.workers,
		'pin_memory':True,
		'persistent_workers':args.workers > 0,
		'prefetch_factor':4 if args.workers > 0 else None}

	train_dloader = DataLoader(
		train_dataset,
		batch_size=hp['batch'],
		drop_last=True, # no smaller last batch -> no recompile/extra CUDA graph
		shuffle=True,
		**loader_kwargs)

	valid_dloader = DataLoader(
		valid_dataset,
		batch_size=hp['batch'],
		drop_last=False,
		shuffle=False,
		**loader_kwargs)

	# FIXED SUBSET FOR DDIM-SAMPLED METRICS: 512 CHIPS EVENLY SPACED OVER THE (SORTED) VALIDATION SET
	sample_idx = np.unique(np.linspace(0,len(valid_dataset)-1,min(512,len(valid_dataset)),dtype=int))
	sample_dloader = DataLoader(
		Subset(valid_dataset,sample_idx.tolist()),
		batch_size=2*hp['batch'], # no gradients -> room for larger batches, fewer sequential DDIM calls
		drop_last=False,
		shuffle=False,
		**loader_kwargs)

	# LRATE SCHEDULER -- LINEAR WARMUP OVER 5 EPOCHS (STEPPED PER BATCH), THEN CONSTANT LR
	warmup_steps = 5*len(train_dloader)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: min(1.0,(step+1)/warmup_steps))
	# LINEAR WARMUP OVER 1000 OPTIMIZER STEPS, THEN CONSTANT LR
	# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: min(1.0,(step+1)/1000))

	# FORWARD + REVERSE DIFFUSION
	diffusion = GaussianDiffusion(model,timesteps=1000,device=device)

	# LOG PATHS/BUFFERS
	log_file_path = f'{args.log_dir}/epochs_{model.model_id:03}.tsv'
	logger        = Logger(log_file_path,n_classes=n_classes)
	best_iou = 0.0
	best_iou_epoch = 0
	# best_dice = 0.0
	# best_dice_epoch = 0
	recent_best = RecentBestTracker(n=2)


	for epoch in range(hp['epochs']):

		# TRAIN & VALIDATE
		start_time = time.perf_counter()
		train_loss = train(model,diffusion,train_dloader,optimizer,scheduler,device,normalize,noise=noise,n_classes=n_classes)
		print(f"Epoch {epoch}: train_loss={train_loss:.5f}")
		valid_loss = validate(model,diffusion,valid_dloader,device,normalize,n_classes=n_classes)
		print(f"Epoch {epoch}: valid_loss={valid_loss:.5f}")
		train_time = time.perf_counter() - start_time
		print(f"Train+validation time: {train_time:.2f} secs.")

		# SAMPLED SEGMENTATION METRICS (SUBSET) -- USE THESE FOR MODEL SELECTION
		sample_metrics = (epoch+1) % 5 == 0
		if sample_metrics:
			sampling_start_time = time.perf_counter()
			ce_loss,valid_cmat = validate_sampling(model,diffusion,sample_dloader,device,normalize,n_classes=n_classes)
			sampling_time = time.perf_counter() - sampling_start_time
			va_metrics = calculate_metrics(valid_cmat.cpu())
		else:
			ce_loss    = float('nan')
			va_metrics = {k:torch.full((n_classes,),float('nan')) for k in ('ppv','tpr','acc','iou','dice')}

		# COLLECT METRICS & RESULTS
		results = {'tloss':train_loss,'vloss':valid_loss,'closs':ce_loss}
		results.update(va_metrics)
		if sample_metrics:
			print(f"CE: {results['closs']:.5f} | IoU: {results['iou'][1]:.5f} | Dice: {results['dice'][1]:.5f}")
			print(f"Sampling time: {sampling_time:.2f} secs.")

		# LOG WHOLE EPOCH
		logger.log(results)

		# SAVE CHECKPOINTS
		if sample_metrics:
			if best_iou < results['iou'][1]:
				best_iou = results['iou'][1]
				chkpt_path = save_checkpoint(args.net_dir,model,optimizer,epoch,'best')
				recent_best.update(chkpt_path)

		print('-'*60)


if __name__ == '__main__':

	args = parse_args()
	train_and_validate(args)
