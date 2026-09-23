import torch
import torchvision
from torchvision import tv_tensors
from PIL import Image
import torchvision.transforms as v1
import torchvision.transforms.v2 as v2
import glob
import os

################################################################################
# CLASSES
################################################################################
class TrainTransform:
	'''
	Class for the augmentation performed during training. Image and label use
	same geometric rotation/flip. Color jitter and blur applied only to bands.
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

		self.intensity = v2.Compose([
			v2.GaussianNoise(mean=0,sigma=0.02*255,clip=False)
		])


	def __call__(self,image,label):
		image, label = self.geometric(image,label)
		image        = self.intensity(image)
		return image, label



class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,chip_dir,n_bands=3,n_labels=2,transform=None,mask_dir=None):

		# GET LIST OF CHIPS/STRINGS
		self.dir        = chip_dir
		self.band_files = sorted(glob.glob(f"{chip_dir}/*_B0X.tif"))
		self.ids        = [i[0:-8] for i in self.band_files]

		# NORMALIZING CONSTANTS
		self.mean = torch.tensor([123.305154,132.731427,131.599954,115.507302]).view(-1,1,1)
		self.std  = torch.tensor([53.223368,51.529520,53.707588,55.610260]).view(-1,1,1)

		# ADJUST ARRAY GEOMETRIES ACCORDING TO NR OF BANDS
		if (n_bands!=3) and (n_bands!=4):
			raise ValueError("Incorrect number of bands in dataloader.")
		if n_bands == 3:
			self.band_func = self.get_rgb
			self.mean = self.mean[0:3]
			self.std  = self.std[0:3]
		if n_bands == 4:
			self.band_func = self.get_vnir

		# ADJUST LABEL INDICES ACCORDING TO NR OF LABELS
		if (n_labels)!=3 and (n_labels!=2):
			raise ValueError("Incorrect number of target labels.")
		if n_labels == 2:
			self.lbl_div = 255
		if n_labels == 3:
			self.lbl_div = 127


		# WEIGHT MASK -- ADDITIONAL IMAGE FOR LOSS
		self.boundary = mask_dir is not None
		if mask_dir is not None:
			self.mask_dir = mask_dir.rstrip('/')

		# TRANSFORMS ONLY FOR TRAIN SET -- AUGMENTATION
		self.train_transform = transform


	def get_rgb(self,idx):
		'''
		Return 3 bands.
		'''
		r,g,b,_ = Image.open(f'{self.ids[idx]}_B0X.tif').split()	
		return Image.merge(mode='RGB',bands=[r,g,b])


	def get_vnir(self,idx):
		'''
		Return 4 bands.
		'''
		return Image.open(f'{self.ids[idx]}_B0X.tif')


	def load_image(self,idx):
		'''
		Return as float32 tensor wrapped in tv_tensor.Image class.
		Values 0-255
		'''
		img = v2.functional.pil_to_tensor(self.band_func(idx))
		img = img.to(torch.float32)
		return tv_tensors.Image(img)


	def load_label(self,idx):
		'''
		Adjust to discrete labels and return as tv_tensors.Mask class.
		Values 0,1
		'''
		lbl = v2.functional.pil_to_tensor(Image.open(f'{self.ids[idx]}_LBL.tif'))
		lbl = torch.squeeze(lbl,0)
		lbl = torch.div(lbl,self.lbl_div,rounding_mode='floor').to(torch.int64)
		return tv_tensors.Mask(lbl)


	def load_label_mask(self,idx):
		basename = os.path.basename(self.ids[idx])
		msk = v2.functional.pil_to_tensor(Image.open(f'{self.mask_dir}/{basename}_MSK.tif'))
		msk = torch.squeeze(msk,0)
		msk = msk.to(torch.float32)
		return msk


	def __len__(self):
		return len(self.ids)


	def __getitem__(self,idx):

		# GET RGB (or RGB+NIR) & LABELS
		image = self.load_image(idx) #tv_tensors.Image object
		label = self.load_label(idx) #tv_tensors.Mask object

		# AUGMENT -- TRAINING
		if self.train_transform:
			image,label = self.train_transform(image,label)

		# NORMALIZE & RETURN
		image = (image - self.mean) / self.std

		# RETURN 2 (OR 3 IMAGES)
		if self.boundary:
			distmap = self.load_label_mask(idx)
			return image,label,distmap
		else:
			return image,label
