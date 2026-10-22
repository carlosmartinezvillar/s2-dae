import os
import numpy as np
import torch
import random
import json

# TODO
class DiffLogger():
	'''
	Class to log the metrics of unguided diffusion pre-training.
	'''
	def __init__(self,path):
		self.path = path

	def log(self,metrics):
		return True

# TODO
class BTLogger():
	'''
	Class to log the metrics of BT.
	'''
	def __init__(self,path):
		self.path = path
		self.header = None

	def log(self,metrics):
		if self.header == None:
			self.header = list(metrics.keys())
			with open(self.path,'w') as fp:
				fp.write('\t'.join(self.header)+'\n')

		row = [f'{v:.5f}' if isinstance(v,float) else str(v) for v in metrics.values()]
		with open(self.path,'a') as fp:
			fp.write('\t'.join(row) + '\n')


class SegmentationLogger():
	def __init__(self,path,n_classes=2):
		'''
		Class to log the segmentation metrics.
		'''
		self.path      = path
		self.n_classes = n_classes
		self.per_epoch = ('tloss','vloss')
		self.per_class = ('acc','tpr','ppv','iou','dice')
		self.header    = None

	def log(self,metrics):
		'''
		metrics: Dict
		'''
		if self.header is None:
			self.header = list(metrics.keys())
			with open(self.path,'w') as fp:
				fp.write('\t'.join(self.header)+'\n')
			# for prefix in per_class:
			# 	header += [f'{prefix}_{c}' for c in range(self.n_classes)]

		# row = [f"{metrics['tloss']:.5f}",f"{metrics['vloss']:.5f}",f"{metrics['closs']:.5f}"]
		# for prefix in self.per_class:
			# row += [f'{metrics[prefix][c]:.5f}' for c in range(self.n_classes)]

		row = [f'{v:.5f}' if isinstance(v,float) else str(v) for v in metrics.values()]

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
	save_path = f'{path}/{tag}_{model.model_id:03}_e{epoch:04}.pth.tar'

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

