import math
import torch
import torch.nn.functional as F
import numpy as np

def mask_to_x0(mask,num_classes):
	'''
	Mask to [B,C,H,W] in range [-1,1]
	'''
	one_hot = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
	return one_hot * 2 - 1


def x0_to_mask(x0):
	'''
	Denoised x0 [B,2,H,W] -> [B,H,W]
	'''
	return x0.argmax(dim=1)


def linear_beta_schedule(timesteps,beta_start=1e-4,beta_end=2e-2):
	return torch.linspace(beta_start,beta_end,timesteps)


def cosine_beta_schedule(timesteps,s=0.008):
	'''
	From Nichol & Dhariwal
	'''
	steps = timesteps+1
	t     = torch.linspace(0,timesteps,steps) / timesteps
	alphas_cumprod = torch.cos((t+s)/(1+s) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
	return torch.clamp(betas,0.0001,0.9999)


class GaussianDiffusion:

	def __init__(self,model,timesteps=1000,ddim_steps=50,device='cpu'):

		self.model     = model
		self.timesteps = timesteps
		self.device    = device

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

		# q(x_{t-1} | x_t, x_0) VAR
		self.posterior_variance = (betas*(1.0-alphas_cumprod_prev)/(1.0-alphas_cumprod)).to(device)

		# DDIM sub-sequence of timesteps
		self.ddim_timesteps = torch.from_numpy(np.linspace(0,timesteps-1,ddim_steps,dtype=int)).to(device)
		self.ddim_alphas_bar = self.alphas_cumprod[self.ddim_timesteps]
		# Prepend a 1.0 for the final step (t=0 requires alpha_bar_{t-1} which is 1.0)
		self.ddim_alphas_bar_prev = torch.cat([torch.ones(1,device=device),self.ddim_alphas_bar[:-1]])


	def broadcast(self,buffer,t,shape):
		'''
		Returns [B,1,1,1] version of 'buffer' tensor to operate on batch.
		'''
		return buffer[t].view(-1, *[1] * (len(shape) - 1))


	def q_sample(self,x0,t,noise):
		'''
		Forward process: x_t ~ q(x_t | x_0)
		'''
		# noise = torch.randn_like(x0)
		sqrt_alphas_cumprod_t           = self.broadcast(self.sqrt_alphas_cumprod,t,x0.shape)
		sqrt_one_minus_alphas_cumprod_t = self.broadcast(self.sqrt_one_minus_alphas_cumprod,t,x0.shape)

		#mean + variance
		return sqrt_alphas_cumprod_t*x0 + sqrt_one_minus_alphas_cumprod_t*noise


	def p_loss(self,x0,cond_image,t):
		'''
		Training loss. MSE of noise versus predicted-noise.
		'''
		noise = torch.randn_like(x0)
		x_t = self.q_sample(x0,t,noise)
		predicted_noise = self.model(x_t,t,cond_image)
		return F.mse_loss(predicted_noise,noise)


	@torch.no_grad()
	def p_sample(self,x_t,t,t_index,cond_image):
		'''
		Single reverse step x_{t-1} ~ p(x_{t-1} | x_t)
		'''
		betas_t                          = self.broadcast(self.betas, t, x_t.shape)
		sqrt_one_minus_alphas_cumprod_t  = self.broadcast(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
		sqrt_recip_alphas_t              = self.broadcast(self.sqrt_recip_alphas, t, x_t.shape)		

		predicted_noise = self.model(x_t,t,cond_image)
		model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

		# Last step
		if t_index==0:
			return model_mean

		# All other in-between steps
		posterior_variance_t = self.broadcast(self.posterior_variance,t,x_t.shape)
		noise = torch.randn_like(x_t)
		return model_mean + torch.sqrt(posterior_variance_t) * noise


	@torch.no_grad()
	def p_sample_loop(self,cond_image,mask_channels=1):
		'''
		Full reverse process: x_T ~ N(0,I) -> x_0
		'''
		B, _, H, W = cond_image.shape
		x_t = torch.randn(B, mask_channels, H, W, device=cond_image.device)

		for t_index in reversed(range(self.timesteps)):
			t   = torch.full((B,), t_index, device=cond_image.device, dtype=torch.long)
			x_t = self.p_sample(x_t,t,t_index,cond_image)

		return x_t


	@torch.no_grad()
	def ddim_sample_loop(self,cond_image,mask_channels=1):
		'''
		Denoising Diffusion Implicit Models (DDIM) Song et al. (2020).
		Reverse process sampling with skipped timesteps for faster validation. 
		'''
		B, _, H, W = cond_image.shape

		# Start with pure noise
		x_t = torch.randn((B,mask_channels,H,W), device=cond_image.device)

		# Loop backwards through substeps
		for i in reversed(range(len(self.ddim_timesteps))):

			# Create a batch-wide tensor for the current actual timestep
			t_tensor = self.ddim_timesteps[i].repeat(B)

			# 1. Predict the noise
			predicted_noise = self.model(x_t, t_tensor, cond_image)

			# 2. Mathematically estimate x_0 (the final clean mask logits)
			alpha_bar      = self.ddim_alphas_bar[i]
			alpha_bar_prev = self.ddim_alphas_bar_prev[i]
			pred_x0 = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)

			# 3. Calculate the direction pointing to x_t-1
			dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise

			# 4. Deterministic jump to the next step in your sub-sequence
			x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

		# 'x_t' final segmentation logits
		return x_t



if __name__ == '__main__':

	# CHECK FORWARD PROCESS ON ONE IMAGE/MASK PAIR
	from dataloader import SentinelDataset
	diffusion = GaussianDiffusion(model=None,timesteps=1000,device='cpu')

	chip_dir  = '../chips_256_sorted/training'
	train_set = SentinelDataset(chip_dir,n_bands=3,n_labels=2,transform=None,mask_dir=None)
	rgb,mask  = train_set[100]                            # [3,H,W], [H,W]

	x0    = mask_to_x0(mask.unsqueeze(0),num_classes=2) # [1,2,H,W]
	noise = torch.randn_like(x0)
	ts    = [0,100,250,500,750,999]

	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(1,len(ts)+1,figsize=(2.5*(len(ts)+1),3))

	# COLUMN 0: INPUT RGB (UN-NORMALIZED), NOT NOISED -- CONDITIONING IMAGE
	img = ((rgb*train_set.std + train_set.mean)/255).clamp(0,1)
	axes[0].imshow(img.permute(1,2,0))
	axes[0].set_title('RGB')

	# REMAINING COLUMNS: ARGMAX MASK OF x_t
	for i,t in enumerate(ts,start=1):
		x_t = diffusion.q_sample(x0,torch.tensor([t]),noise)
		acc = (x0_to_mask(x_t) == mask).float().mean()  # fraction of pixels whose class survives the noise
		print(f"t={t:4d}  signal={diffusion.sqrt_alphas_cumprod[t]:.3f}  noise={diffusion.sqrt_one_minus_alphas_cumprod[t]:.3f}  mask_acc={acc:.3f}")

		axes[i].imshow(x0_to_mask(x_t)[0],cmap='gray',vmin=0,vmax=1)
		axes[i].set_title(f't={t}  acc={acc:.2f}')

	for ax in axes.flat:
		ax.axis('off')
	plt.tight_layout()
	plt.savefig('fig/forward_diffusion.png',dpi=120)
	print("saved fig/forward_diffusion.png")
