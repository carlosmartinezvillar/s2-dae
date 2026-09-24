'''
Faster alternative to dataloader.SentinelDataset. Self-contained: does not import dataloader.py.
  - build_cache():        decodes all chips of a split ONCE into uint8 .npy files
                          (images [N,C,H,W], labels [N,H,W]) that are memory-mapped at train time.
  - FastSentinelDataset:  reads a chip as a slice of the memory-mapped cache (no TIFF decode)
                          and returns the RAW uint8 image -- 4x less data to copy to the GPU.
  - FastTrainTransform:   training augmentation, geometric part (flips/rotations) on uint8 in the
                          dataloader workers.
  - GPUGaussianNoise:     training augmentation, noise part on the GPU (float-only op).
  - GPUNormalize:         (image - mean) / std on the GPU -- for training, validation and testing.

Usage:
	python fast_dataloader.py --chip-dir ../chips_256_sorted/training   --cache-dir ../chips_256_cache/training
	python fast_dataloader.py --chip-dir ../chips_256_sorted/validation --cache-dir ../chips_256_cache/validation

	train_set = FastSentinelDataset('../chips_256_cache/training',transform=FastTrainTransform())
	valid_set = FastSentinelDataset('../chips_256_cache/validation')
	noise     = GPUGaussianNoise()
	normalize = GPUNormalize(train_set.mean,train_set.std,device)

	# TRAINING
	for rgb,lbl in train_loader:
		rgb = normalize(noise(rgb.to(device,non_blocking=True)))
		lbl = lbl.to(device,non_blocking=True)

	# VALIDATION / TESTING
	for rgb,lbl in valid_loader:
		rgb = normalize(rgb.to(device,non_blocking=True))
		lbl = lbl.to(device,non_blocking=True)
'''

import argparse
import glob
import json
import os
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from torchvision import tv_tensors


################################################################################
# CONSTANTS
################################################################################
# PER-BAND NORMALIZING CONSTANTS (R,G,B,NIR) ON THE 0-255 SCALE
BAND_MEAN = [123.305154,132.731427,131.599954,115.507302]
BAND_STD  = [53.223368,51.529520,53.707588,55.610260]

# LABEL TIFF VALUES -> CLASS INDICES: floor(value / LABEL_DIV[n_labels])
LABEL_DIV = {2:255,3:127}


################################################################################
# CACHE
################################################################################
def build_cache(chip_dir,cache_dir,n_bands=3,n_labels=2):
	'''
	Decode every chip in 'chip_dir' (*_B0X.tif bands, *_LBL.tif labels) and write them to
	'cache_dir' as uint8 .npy files, plus a meta.json with the normalizing constants and chip ids.
	n_bands=3 keeps R,G,B of the 4-band chips; n_bands=4 keeps R,G,B,NIR.
	Written through open_memmap, so the full split never has to fit in RAM.
	'''
	assert n_bands in (3,4), "Incorrect number of bands."
	assert n_labels in LABEL_DIV, "Incorrect number of target labels."

	band_files = sorted(glob.glob(f"{chip_dir}/*_B0X.tif"))
	ids        = [f[0:-8] for f in band_files]
	N          = len(ids)
	assert N > 0, f"No chips found in {chip_dir}"

	# SHAPES FROM FIRST CHIP
	H,W = np.array(Image.open(band_files[0])).shape[0:2]

	os.makedirs(cache_dir,exist_ok=True)
	images = np.lib.format.open_memmap(f'{cache_dir}/images.npy',mode='w+',dtype=np.uint8,shape=(N,n_bands,H,W))
	labels = np.lib.format.open_memmap(f'{cache_dir}/labels.npy',mode='w+',dtype=np.uint8,shape=(N,H,W))

	for idx,chip_id in enumerate(ids):
		bands = np.array(Image.open(f'{chip_id}_B0X.tif'))          # [H,W,4] uint8
		images[idx] = bands[:,:,0:n_bands].transpose(2,0,1)         # [n_bands,H,W]
		labels[idx] = np.array(Image.open(f'{chip_id}_LBL.tif')) // LABEL_DIV[n_labels]
		if (idx+1) % 1000 == 0 or idx+1 == N:
			print(f"  {idx+1}/{N} chips cached")

	images.flush()
	labels.flush()

	meta = {
		'chip_dir':os.path.abspath(chip_dir),
		'n_bands':n_bands,
		'n_labels':n_labels,
		'mean':BAND_MEAN[0:n_bands],
		'std':BAND_STD[0:n_bands],
		'ids':[os.path.basename(i) for i in ids]}
	with open(f'{cache_dir}/meta.json','w') as fp:
		json.dump(meta,fp)

	print(f"Cache written to {cache_dir}: images {images.shape}, labels {labels.shape}")


################################################################################
# TRANSFORMS
################################################################################
class FastTrainTransform:
	'''
	Training augmentation, geometric part: same random flip/rotation on image and label.
	Works on uint8 tv_tensors, so it runs in the dataloader workers before the transfer.
	Rotations are restricted to multiples of 90 degrees (no interpolation, no padding).
	The noise part of the augmentation is GPUGaussianNoise, applied on the GPU.
	'''
	def __init__(self):
		self.geometric = v2.Compose([
			v2.RandomHorizontalFlip(p=0.5),
			v2.RandomVerticalFlip(p=0.5),
			v2.RandomChoice([
				v2.RandomRotation([0,0]),
				v2.RandomRotation([90,90]),
				v2.RandomRotation([180,180]),
				v2.RandomRotation([270,270])
			])
		])

	def __call__(self,image,label):
		return self.geometric(image,label)


class GPUGaussianNoise:
	'''
	Training augmentation, noise part: additive Gaussian noise on the 0-255 scale (unclipped),
	applied to a uint8/float [B,C,H,W] batch on the GPU BEFORE GPUNormalize.
	Training batches only.
	'''
	def __init__(self,mean=0.0,sigma=0.02*255):
		self.mean  = mean
		self.sigma = sigma

	def __call__(self,x):
		x = x.float()
		return x + torch.randn_like(x) * self.sigma + self.mean


################################################################################
# DATASET
################################################################################
class FastSentinelDataset(torch.utils.data.Dataset):
	'''
	Reads chips from a cache written by build_cache().
	Returns (image,label): image uint8 [C,H,W] (NOT normalized -- use GPUNormalize on the GPU),
	label int64 [H,W] class indices.

	transform: optional callable (image,label) -> (image,label) on uint8 tv_tensors,
	           e.g. FastTrainTransform() for the training set; None for validation/testing.
	'''
	def __init__(self,cache_dir,transform=None):
		with open(f'{cache_dir}/meta.json','r') as fp:
			meta = json.load(fp)

		self.cache_dir = cache_dir
		self.n_bands   = meta['n_bands']
		self.n_labels  = meta['n_labels']
		self.ids       = meta['ids']
		self.mean      = torch.tensor(meta['mean']).view(-1,1,1)
		self.std       = torch.tensor(meta['std']).view(-1,1,1)
		self.train_transform = transform

		# OPENED LAZILY IN EACH WORKER (memmaps are not shared across forked workers)
		self.images = None
		self.labels = None
		self.length = len(self.ids)

	def _open(self):
		self.images = np.load(f'{self.cache_dir}/images.npy',mmap_mode='r')
		self.labels = np.load(f'{self.cache_dir}/labels.npy',mmap_mode='r')

	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		if self.images is None:
			self._open()

		image = tv_tensors.Image(torch.from_numpy(np.array(self.images[idx]))) # uint8 [C,H,W], copied out of the memmap
		label = tv_tensors.Mask(torch.from_numpy(self.labels[idx].astype(np.int64)))

		if self.train_transform:
			image,label = self.train_transform(image,label)

		return image,label


################################################################################
# GPU NORMALIZATION
################################################################################
class GPUNormalize:
	'''
	[B,C,H,W] batch on device (uint8, or float after GPUGaussianNoise) -> float32 (x - mean) / std.
	Needed for every split: training, validation and testing.
	'''
	def __init__(self,mean,std,device):
		self.mean = mean.view(1,-1,1,1).to(device)
		self.std  = std.view(1,-1,1,1).to(device)

	def __call__(self,x):
		return (x.float() - self.mean) / self.std


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Build a uint8 .npy cache of a chip directory for FastSentinelDataset.')
	parser.add_argument('--chip-dir',required=True,help='Directory with *_B0X.tif / *_LBL.tif chips.')
	parser.add_argument('--cache-dir',required=True,help='Output directory for images.npy, labels.npy, meta.json.')
	parser.add_argument('--bands',type=int,default=3,choices=[3,4])
	parser.add_argument('--labels',type=int,default=2,choices=[2,3])
	args = parser.parse_args()

	build_cache(args.chip_dir,args.cache_dir,n_bands=args.bands,n_labels=args.labels)
