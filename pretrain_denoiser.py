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

def train(model,gpu_aug,diffusion,loader,optimizer,scheduler,device):
	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	model.train()

	for vnir,mask in loader:

		# TO DEV & NORM
		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)
		vnir = gpu_aug(vnir)

		# GET X0 & RANDOM TIMESTEPS
		x0 = mask_to_x0(mask,2)
		t  = torch.randint(0,diffusion.timesteps,(vnir.shape[0],),device=device,dtype=torch.long)

		# FORWARD
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			loss = diffusion.p_loss(x0,vnir,t)	

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
		optimizer.step()		

		loss_sum += loss.detach() * vnir.size(0)
		n_samples += vnir.size(0)

	scheduler.step()

	return (loss_sum/samp_sum).item() # >>>> CPU-GPU SYNC <<<<


if __name__ == '__main__':
	pass