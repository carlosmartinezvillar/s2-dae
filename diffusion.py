'''
A diffusion class to keep
* forward process
* reverse step
* reverse process
* 
'''
import math
import torch
import torch.nn.functional as F
import numpy as np


################################################################################
# CLASS HELPERS
################################################################################
def mask_channels(num_classes):
	'''
	Channels of x0 for a given number of classes. Binary masks use a single
	channel (a one-hot pair is redundant: c0 = -c1); multi-class uses one-hot.
	'''
	return 1 if num_classes == 2 else num_classes


def mask_to_x0(mask,num_classes):
	'''
	Mask [B,H,W] to [B,C,H,W] in range [-1,1], C = mask_channels(num_classes)
	'''
	if num_classes == 2:
		return mask.float().unsqueeze(1) * 2 - 1
	one_hot = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
	return one_hot * 2 - 1


def x0_to_mask(x0):
	'''
	Denoised x0 [B,C,H,W] -> [B,H,W]. Threshold at 0 if C == 1, else argmax.
	'''
	if x0.shape[1] == 1:
		return (x0[:,0] > 0).long()
	return x0.argmax(dim=1)


def linear_beta_schedule(timesteps,beta_start=1e-4,beta_end=2e-2):
	return torch.linspace(beta_start,beta_end,timesteps)


def cosine_beta_schedule(timesteps,s=0.008,max_beta=0.999):
	'''
	From Nichol & Dhariwal. max_beta=0.999 as in the paper: the cosine alpha_bar hits 0 at t=T,
	so the last beta is clamped (0.9999 left alpha_bar[T-1]=2.4e-10 and a ~1000x first reverse step).
	'''
	steps = timesteps+1
	t     = torch.linspace(0,timesteps,steps) / timesteps
	alphas_cumprod = torch.cos((t+s)/(1+s) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
	return torch.clamp(betas,0.0001,max_beta)

################################################################################
# MAIN CLASS
################################################################################
class GaussianDiffusion:

	#################### SET SCHEDULE ####################
	def __init__(self,model,timesteps=1000,ddim_steps=50,device='cpu',guided=True,x0_min=-1.0,x0_max=1.0):
		'''
		x0_min/x0_max: range of valid x0, the predicted x0 is clipped to it during sampling.
		Floats apply to every channel (masks in [-1,1]); a length-C sequence/tensor gives
		per-channel bounds (normalized VNIR bands: (0-MEAN)/STDD and (255-MEAN)/STDD).
		'''
		self.model     = model
		self.guided    = guided # False: model(x_t,t), x0 HAS THE SHAPE OF THE VNIR BATCH
		self.timesteps = timesteps
		self.device    = device

		# x0 RANGE AS [1,C,1,1] (C=1 FOR SCALARS, BROADCASTS OVER ANY NR OF CHANNELS)
		self.x0_min = torch.as_tensor(x0_min,dtype=torch.float32,device=device).reshape(1,-1,1,1)
		self.x0_max = torch.as_tensor(x0_max,dtype=torch.float32,device=device).reshape(1,-1,1,1)
		assert self.x0_min.shape == self.x0_max.shape, "x0_min and x0_max need the same nr of channels"
		assert (self.x0_min < self.x0_max).all(), "x0_min must be below x0_max"

		betas = cosine_beta_schedule(timesteps)
		alphas = 1.0 - betas
		alphas_cumprod = torch.cumprod(alphas,dim=0)
		alphas_cumprod_prev = F.pad(alphas_cumprod[:-1],(1,0),value=1.0)

		# BUFFERS
		self.betas = betas.to(device) 
		self.alphas_cumprod = alphas_cumprod.to(device)
		self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)
		self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
		self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)

		# q(x_{t-1} | x_t, x_0) MEAN = coef1*x0 + coef2*x_t, AND VAR
		self.posterior_mean_coef1 = (betas*torch.sqrt(alphas_cumprod_prev)/(1.0-alphas_cumprod)).to(device)
		self.posterior_mean_coef2 = ((1.0-alphas_cumprod_prev)*torch.sqrt(alphas)/(1.0-alphas_cumprod)).to(device)
		self.posterior_variance = (betas*(1.0-alphas_cumprod_prev)/(1.0-alphas_cumprod)).to(device)

		# DDIM sub-sequence of timesteps
		self.ddim_timesteps = torch.from_numpy(np.linspace(0,timesteps-1,ddim_steps,dtype=int)).to(device)
		self.ddim_alphas_bar = self.alphas_cumprod[self.ddim_timesteps]
		# Prepend a 1.0 for the final step (t=0 requires alpha_bar_{t-1} which is 1.0)
		self.ddim_alphas_bar_prev = torch.cat([torch.ones(1,device=device),self.ddim_alphas_bar[:-1]])


	#################### TRAIN ONLY ####################
	def q_sample(self,x0,t,noise):
		'''
		Forward process: x_t ~ q(x_t | x_0)
		'''
		# noise = torch.randn_like(x0)
		sqrt_alphas_cumprod_t           = self.broadcast(self.sqrt_alphas_cumprod,t,x0.shape)
		sqrt_one_minus_alphas_cumprod_t = self.broadcast(self.sqrt_one_minus_alphas_cumprod,t,x0.shape)

		#mean + variance
		return sqrt_alphas_cumprod_t*x0 + sqrt_one_minus_alphas_cumprod_t*noise


	def p_loss(self,x0,t,cond_image=None):
		'''
		Training loss. MSE of noise versus predicted-noise.
		Guided: p_loss(x0,t,cond_image). Unguided: p_loss(x0,t).
		'''
		noise = torch.randn_like(x0)
		x_t = self.q_sample(x0,t,noise)
		predicted_noise = self._predict_noise(x_t,t,cond_image)
		return F.mse_loss(predicted_noise,noise)


	#################### SAMPLE AND TRAIN ####################
	def broadcast(self,buffer,t,shape):
		'''
		Returns [B,1,1,1] version of 'buffer' tensor to operate on batch.
		'''
		return buffer[t].view(-1, *[1] * (len(shape) - 1))


	def _predict_noise(self,x_t,t,cond_image=None):
		'''
		Guided: model(x_t,t,cond_image). Unguided: model(x_t,t).
		Cast to float32: under bf16 autocast the model returns bf16, the sampling math runs in float32.
		'''
		if self.guided:
			return self.model(x_t,t,cond_image).float()
		return self.model(x_t,t).float()


	#################### SAMPLING ONLY ####################
	def _clip_x0(self,x0):
		'''
		Clip a predicted x0 [B,C,H,W] to the valid data range, per channel.
		'''
		C = self.x0_min.shape[1]
		assert C in (1,x0.shape[1]), f"x0 range has {C} channels, x0 has {x0.shape[1]}"
		return torch.clamp(x0,min=self.x0_min,max=self.x0_max)


	@torch.no_grad()
	def p_sample(self,x_t,t,t_index,cond_image=None,generator=None):
		'''
		Single reverse step x_{t-1} ~ p(x_{t-1} | x_t). Mean of q(x_{t-1}|x_t,x0) at the clipped x0 estimate.
		'''
		alpha_bar_t = self.broadcast(self.alphas_cumprod, t, x_t.shape)
		coef1_t     = self.broadcast(self.posterior_mean_coef1, t, x_t.shape)
		coef2_t     = self.broadcast(self.posterior_mean_coef2, t, x_t.shape)

		predicted_noise = self._predict_noise(x_t,t,cond_image)

		# x0 ESTIMATE, CLIPPED TO THE DATA RANGE (NEAR t=T THE DIVISION BY sqrt(alpha_bar) AMPLIFIES NOISE ERRORS)
		pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
		pred_x0 = self._clip_x0(pred_x0)

		model_mean = coef1_t * pred_x0 + coef2_t * x_t

		# Last step
		if t_index==0:
			return model_mean

		# All other in-between steps
		posterior_variance_t = self.broadcast(self.posterior_variance,t,x_t.shape)
		noise = torch.randn(x_t.shape,generator=generator,device=x_t.device)
		return model_mean + torch.sqrt(posterior_variance_t) * noise


	@torch.no_grad()
	def p_sample_loop(self,x0,cond_image=None,generator=None):
		'''
		Full reverse process: x_T ~ N(0,I) -> x_0
		x0: tensor with the shape of the denoised target; only its shape and device are used for x_T
		(unguided: the VNIR batch; guided: the mask as x0). cond_image: guided models only.
		generator must live on x0.device.
		'''
		x_t = torch.randn(x0.shape,generator=generator,device=x0.device)
		B   = x0.shape[0]

		for t_index in reversed(range(self.timesteps)):
			t   = torch.full((B,), t_index, device=x0.device, dtype=torch.long)
			x_t = self.p_sample(x_t,t,t_index,cond_image,generator)

		return x_t


	@torch.no_grad()
	def ddim_sample_loop(self,x0,cond_image=None,generator=None):
		'''
		Denoising Diffusion Implicit Models (DDIM) Song et al. (2020).
		Reverse process sampling with skipped timesteps for faster validation.
		x0: tensor with the shape of the denoised target; only its shape and device are used for x_T
		(unguided: the VNIR batch; guided: the mask as x0). cond_image: guided models only.
		generator must live on x0.device.
		'''
		# Start with pure noise
		x_t = torch.randn(x0.shape,generator=generator,device=x0.device)
		B   = x0.shape[0]

		# Loop backwards through substeps
		for i in reversed(range(len(self.ddim_timesteps))):

			# Create a batch-wide tensor for the current actual timestep
			t_tensor = self.ddim_timesteps[i].repeat(B)

			# 1. Predict the noise
			predicted_noise = self._predict_noise(x_t,t_tensor,cond_image)

			# 2. Estimate x_0, CLIPPED TO THE DATA RANGE, AND THE NOISE CONSISTENT WITH THE CLIPPED x_0
			alpha_bar      = self.ddim_alphas_bar[i]
			alpha_bar_prev = self.ddim_alphas_bar_prev[i]
			pred_x0 = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
			pred_x0 = self._clip_x0(pred_x0)
			eps     = (x_t - torch.sqrt(alpha_bar) * pred_x0) / torch.sqrt(1 - alpha_bar) # NOISE CONSISTENT WITH THE CLIPPED x_0

			# 3. Calculate the direction pointing to x_t-1
			dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps

			# 4. Deterministic jump to the next step in your sub-sequence
			x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

		# 'x_t' final x0 estimate: mask logits (guided) or normalized bands (unguided)
		return x_t



if __name__ == '__main__':

	# CHECK FORWARD PROCESS ON ONE CHIP: MASK (GUIDED TARGET) AND VNIR BANDS (UNGUIDED TARGET)
	from dataloader import SentinelDataset
	diffusion = GaussianDiffusion(model=None,timesteps=1000,device='cpu')

	chip_dir  = '../chips_256_sorted/training'
	train_set = SentinelDataset(chip_dir,n_bands=4,n_labels=2,transform=None,mask_dir=None)
	vnir,mask = train_set[100]                            # [4,H,W] NORMALIZED, [H,W]
	vnir      = vnir.as_subclass(torch.Tensor)            # DROP tv_tensors WRAPPER

	x0_mask = mask_to_x0(mask.unsqueeze(0),num_classes=2) # [1,1,H,W]
	x0_vnir = vnir.unsqueeze(0)                           # [1,4,H,W]
	ts      = [0,100,250,500,750,999]

	def to_rgb(x):
		# NORMALIZED BANDS -> [0,1] RGB FROM THE FIRST 3 BANDS
		return ((x[:3]*train_set.std[:3] + train_set.mean[:3])/255).clamp(0,1).permute(1,2,0)

	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(2,len(ts)+1,figsize=(2.5*(len(ts)+1),5.5))

	# COLUMN 0: CLEAN INPUTS
	axes[0,0].imshow(to_rgb(vnir))
	axes[0,0].set_title('VNIR (RGB)')
	axes[1,0].imshow(mask,cmap='gray',vmin=0,vmax=1)
	axes[1,0].set_title('mask')

	# REMAINING COLUMNS: NOISED BANDS (ROW 0) AND THRESHOLDED NOISED MASK (ROW 1)
	for i,t in enumerate(ts,start=1):
		tt = torch.tensor([t])
		x_t_vnir = diffusion.q_sample(x0_vnir,tt,torch.randn_like(x0_vnir))
		x_t_mask = diffusion.q_sample(x0_mask,tt,torch.randn_like(x0_mask))
		acc = (x0_to_mask(x_t_mask) == mask).float().mean()  # fraction of pixels whose class survives the noise
		print(f"t={t:4d}  signal={diffusion.sqrt_alphas_cumprod[t]:.3f}  noise={diffusion.sqrt_one_minus_alphas_cumprod[t]:.3f}  mask_acc={acc:.3f}")

		axes[0,i].imshow(to_rgb(x_t_vnir[0]))
		axes[0,i].set_title(f't={t}')
		axes[1,i].imshow(x0_to_mask(x_t_mask)[0],cmap='gray',vmin=0,vmax=1)
		axes[1,i].set_title(f't={t}  acc={acc:.2f}')

	for ax in axes.flat:
		ax.axis('off')
	plt.tight_layout()
	plt.savefig('fig/forward_diffusion.png',dpi=120)
	print("saved fig/forward_diffusion.png")
