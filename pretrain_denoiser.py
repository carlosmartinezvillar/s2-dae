'''
Pretrain an unguided denoiser (models.UnconditionedDenoiser) on the 4 VNIR bands.
The model learns to predict the noise added to normalized bands at a random step t.
Its weights later initialize the diffusion segmentation model (diff_train).

UNGUIDED: GaussianDiffusion(...,guided=False); the model is called as model(x_t,t).
'''

import os
import glob
import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import argparse

import dataloader
import utils
import models
from diffusion import GaussianDiffusion

LOG_DIR   = None
MODEL_DIR = None

# SCORE COLUMNS LOGGED EVERY EPOCH (NaN ON EPOCHS WITHOUT SAMPLING) -- MUST MATCH sample_and_score()
SCORE_KEYS = ('fid','kid','kid_std','w1_0','w1_1','w1_2','w1_3','hf_0','hf_1','hf_2','hf_3','lpips','lpips_ratio')

# FIXED VALIDATION RNG -- SAME t AND NOISE EVERY EPOCH, SO VALIDATION MSE IS COMPARABLE ACROSS EPOCHS
VAL_SEED = 0

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

	# OPTIONAL
	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--num-workers',required=False,type=int,default=4,help='Set num_workers args.')
	optional.add_argument('--device',required=False,type=int,default=0,help='Override default GPU id 0')

	# LOAD
	args = parser.parse_args()

	# CHECK REQUIRED
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"

	# CHECK OPTIONAL
	assert args.device >= 0, f"Got negative arg for GPU id {args.device}"
	if args.device > 0:
		assert args.device < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."

	# SET GLOBAL VARS -- skip passing as args in train_and_validate()
	global LOG_DIR
	global MODEL_DIR
	LOG_DIR = args.log_dir
	MODEL_DIR = args.net_dir

	# RETURN
	return args


def calculate_fid(mu_r,sigma_r,mu_g,sigma_g):
	'''
	Calculate FID.
	'''
	diff = mu_r - mu_g

	# Tr((Σ_r Σ_g)^½) VIA THE SYMMETRIC FORM Σ_r^½ Σ_g Σ_r^½ (STABLE, EIGENVALUES REAL)
	evals,evecs = torch.linalg.eigh(sigma_r)
	sqrt_r      = evecs @ torch.diag(evals.clamp(min=0).sqrt()) @ evecs.T
	tr_covmean  = torch.linalg.eigvalsh(sqrt_r @ sigma_g @ sqrt_r).clamp(min=0).sqrt().sum()

	return (diff @ diff + torch.trace(sigma_r) + torch.trace(sigma_g) - 2*tr_covmean).item()


def calculate_kid(feat_r,feat_g,n_subsets=10,subset_size=1000,seed=0):
	'''
	Kernel Inception Distance (Binkowski et al. 2018) on any features [N,C]: unbiased MMD^2 with the
	cubic polynomial kernel k(x,y) = (x.y/C + 1)^3, over n_subsets random subsets. Returns (mean,std).
	Unbiased for any subset size, so unlike FID it does not need N >> C.
	'''
	C = feat_r.shape[1]
	m = min(subset_size,feat_r.shape[0],feat_g.shape[0])
	generator = torch.Generator().manual_seed(seed) # CPU INDICES -- SAME REAL SUBSETS EVERY CALL

	mmds = []
	for _ in range(n_subsets):
		x = feat_r[torch.randperm(feat_r.shape[0],generator=generator)[:m].to(feat_r.device)]
		y = feat_g[torch.randperm(feat_g.shape[0],generator=generator)[:m].to(feat_g.device)]

		k_xx = (x @ x.T / C + 1) ** 3
		k_yy = (y @ y.T / C + 1) ** 3
		k_xy = (x @ y.T / C + 1) ** 3

		# UNBIASED: WITHIN-SET TERMS SKIP THE DIAGONAL (A SAMPLE WITH ITSELF)
		mmd = (k_xx.sum() - k_xx.diagonal().sum()) / (m*(m-1)) \
			+ (k_yy.sum() - k_yy.diagonal().sum()) / (m*(m-1)) \
			- 2 * k_xy.mean()
		mmds.append(mmd)

	mmds = torch.stack(mmds)
	return mmds.mean().item(),mmds.std().item()


def calculate_band_wasserstein(hist_r,hist_g):
	'''
	Per-band 1-D Wasserstein-1 distance between pixel-value distributions, from histograms [bands,256]
	over 0..255 (1 DN bins). W1 = sum |CDF_r - CDF_g| * bin width. Returns a list, one value per band, in DN.
	'''
	cdf_r = hist_r.cumsum(dim=1) / hist_r.sum(dim=1,keepdim=True)
	cdf_g = hist_g.cumsum(dim=1) / hist_g.sum(dim=1,keepdim=True)
	return (cdf_r - cdf_g).abs().sum(dim=1).tolist() # BIN WIDTH 1 DN


def calculate_lpips(encoder,x_a,x_b):
	'''
	LPIPS-style distance between paired images in the BT encoder's feature space, one value per pair [B].
	x_a,x_b: normalized bands [B,4,H,W] (all 4 bands). Layers: the 4 skips + the bottleneck. Per layer,
	each pixel's feature vector is scaled to unit length across channels, then the squared difference is
	summed over channels and averaged over pixels; layers are summed (unweighted LPIPS: the learned
	per-layer weights of LPIPS exist only for ImageNet AlexNet/VGG).
	'''
	with torch.autocast(device_type=x_a.device.type,dtype=torch.bfloat16,enabled=True):
		skips_a,out_a = encoder(x_a)
		skips_b,out_b = encoder(x_b)

	dist = torch.zeros(x_a.shape[0],device=x_a.device)
	for f_a,f_b in zip(skips_a+[out_a],skips_b+[out_b]):
		f_a  = F.normalize(f_a.float(),dim=1,eps=1e-10) # UNIT LENGTH ACROSS CHANNELS, PER PIXEL
		f_b  = F.normalize(f_b.float(),dim=1,eps=1e-10)
		dist = dist + ((f_a - f_b)**2).sum(dim=1).mean(dim=(1,2))
	return dist

@torch.no_grad()
def validation_set_statistics(encoder,dataloader,gpu_norm,device,n_pairs=1024,seed=0):
	'''
	Reference statistics of the entire validation set, computed once:
	  feats [N,C]       pooled encoder features (KID), mu [C] and sigma [C,C] (FID)
	  hist  [bands,256] pixel-value histograms in DN (band Wasserstein)
	  grad  [bands]     mean |horizontal/vertical difference| in DN (high-frequency content)
	  lpips             mean BT-feature LPIPS over n_pairs random pairs of distinct chips (diversity reference)
	'''
	encoder.eval()

	feats    = []
	hist     = None
	grad_sum = None
	n_diff   = 0

	for vnir,mask in dataloader:

		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)

		# PIXEL STATISTICS ON THE RAW BANDS (uint8 DN)
		dn = vnir.float()
		if hist is None:
			hist     = torch.zeros(dn.shape[1],256,dtype=torch.float64,device=device)
			grad_sum = torch.zeros(dn.shape[1],dtype=torch.float64,device=device)
		for b in range(dn.shape[1]):
			hist[b] += torch.histc(dn[:,b],bins=256,min=-0.5,max=255.5).double()
		grad_sum += (dn[:,:,1:,:]-dn[:,:,:-1,:]).abs().sum(dim=(0,2,3)).double() # VERTICAL
		grad_sum += (dn[:,:,:,1:]-dn[:,:,:,:-1]).abs().sum(dim=(0,2,3)).double() # HORIZONTAL
		n_diff   += dn.shape[0] * ((dn.shape[2]-1)*dn.shape[3] + dn.shape[2]*(dn.shape[3]-1))

		# ENCODER FEATURES ON THE NORMALIZED BANDS
		vnir,_ = gpu_norm(vnir,mask)
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			skips,features = encoder(vnir)              #_,[B,C,16,16]
			vector = features.mean(dim=(2,3)).double() #[B,C]
		feats.append(vector)

	feats = torch.cat(feats) #[N,C]
	mu    = feats.mean(dim=0)
	sigma = torch.cov(feats.T)

	# LPIPS REFERENCE: RANDOM PAIRS OF DISTINCT CHIPS FROM THE WHOLE SET (FIXED SEED). NOT WITHIN LOADER
	# BATCHES: THE VALIDATION SET IS SORTED BY CHIP ID, SO NEIGHBOURS COME FROM THE SAME TILE AND LOOK ALIKE
	bands   = dataloader.dataset.bands # uint8 [N,4,H,W] IN RAM
	n_pairs = min(n_pairs,len(bands)//2)
	idx     = torch.randperm(len(bands),generator=torch.Generator().manual_seed(seed))[:2*n_pairs]
	dists   = []
	for i in range(0,n_pairs,dataloader.batch_size):
		j     = min(i+dataloader.batch_size,n_pairs)
		x_a   = (bands[idx[i:j]].to(device).float() - gpu_norm.mean) / gpu_norm.stdd # NORMALIZED, AS THE ENCODER EXPECTS
		x_b   = (bands[idx[n_pairs+i:n_pairs+j]].to(device).float() - gpu_norm.mean) / gpu_norm.stdd
		dists.append(calculate_lpips(encoder,x_a,x_b))
	lpips_r = torch.cat(dists).mean().item()

	return {'feats':feats,'mu':mu,'sigma':sigma,'hist':hist,'grad':grad_sum/n_diff,'lpips':lpips_r}


def load_encoder_checkpoint(bt_id,bt_run,epoch=None):
	'''
	Load the frozen BT encoder used by the metrics (fid, kid, lpips).
	Reads {MODEL_DIR}/bt_{bt_id:03}_e{epoch:04}.pth.tar, latest epoch if epoch is None (BT pretraining
	saves only its last epoch; k8s/diff_pretrain.sh copies it from /model_pvc/<bt_run>).
	The architecture comes from row bt_id of ./hpo/{bt_run}.json, so the strict load matches the checkpoint.
	'''
	assert bt_id is not None, "bt_id not set: rerun hpo/experiments.py set_experiments_diff_pretrain(bt_id=<best bt_probe encoder>)"

	# BT ARCHITECTURE FROM ITS PARAMS ROW
	bt_path = f'./hpo/{bt_run}.json'
	with open(bt_path,'r') as fp:
		bt_rows = {row['id']:row for row in map(json.loads,filter(str.strip,fp))}
	assert bt_id in bt_rows, f"BT id {bt_id} not in {bt_path}"
	bt_hp = bt_rows[bt_id]

	# FIND CHECKPOINT
	epoch_str = '*' if epoch is None else f'{epoch:04}'
	paths = sorted(glob.glob(f"{MODEL_DIR}/bt_{bt_id:03}_e{epoch_str}.pth.tar"))
	assert paths, f"No BT checkpoint bt_{bt_id:03}_e{epoch_str}.pth.tar in {MODEL_DIR}"
	path = paths[-1] # ZERO-PADDED EPOCHS, SO LAST IS LATEST
	checkpoint = torch.load(path,map_location='cpu',weights_only=True)

	# REBUILD AS BarlowTwins, LOAD STRICTLY (CATCHES ARCHITECTURE MISMATCH)
	bt_net = models.BarlowTwins(
		model_id=bt_id,
		init_chan=bt_hp['init_chan'],
		cnn_layers=bt_hp['cnn_layers'],
		proj_dim=bt_hp['proj_dim'],
		encoder=bt_hp.get('encoder','cnn')
	)
	bt_net.load_state_dict(checkpoint['model_state_dict'],strict=True)
	print(f"Loaded metric encoder from {path} (epoch {checkpoint['epoch']}, {bt_hp.get('encoder','cnn')}, init_chan {bt_hp['init_chan']})")

	# KEEP ENCODER, DROP PROJECTOR; FROZEN
	encoder = bt_net.encoder
	encoder.requires_grad_(False)
	return encoder.eval()


def train(model,gpu_aug,diffusion,loader,optimizer,scheduler,device):
	'''
	Train unguided model on VNIR bands. Noise-prediction (MSE) loss.
	'''
	model.train()

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	for vnir,mask in loader:

		# TO GPU, AUGMENT & NORMALIZE -- THE MASK IS NOT USED
		vnir = vnir.to(device,non_blocking=True)
		mask = mask.to(device,non_blocking=True)
		x0,_ = gpu_aug(vnir,mask) # NORMALIZED BANDS ARE x0

		# RANDOM TIMESTEPS
		t = torch.randint(0,diffusion.timesteps,(x0.shape[0],),device=device,dtype=torch.long)

		# FORWARD -- NOISE x0 TO x_t, PREDICT THE NOISE
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			# loss = diffusion.p_loss(x0,t) #--> Change to explicit/readable

			noise = torch.randn_like(x0)

			x_t = diffusion.q_sample(x0,t,noise)

			predicted_noise = model(x_t,t).float()

			loss = F.mse_loss(predicted_noise,noise)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
		optimizer.step()

		# LOSS IS A BATCH MEAN -- WEIGHT BY BATCH SIZE FOR A PER-SAMPLE MEAN
		loss_sum  += loss.detach() * x0.size(0)
		n_samples += x0.size(0)

	scheduler.step()

	# SYNC
	return {'tloss':(loss_sum/n_samples).item()}


@torch.no_grad()
def validate(model,gpu_norm,diffusion,loader,device):
	'''
	Noise-prediction (MSE) loss over the full validation set. t and noise are drawn
	from a fixed seed, so every epoch is scored on the same (x0,t,noise) triples.
	'''
	model.eval()

	loss_sum  = torch.zeros(1,device=device)
	n_samples = torch.zeros(1,device=device)

	# FORK THE RNG SO THE FIXED SEED DOES NOT RESET THE TRAINING RNG STREAM
	with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
		torch.manual_seed(VAL_SEED)

		for vnir,mask in loader:

			vnir = vnir.to(device,non_blocking=True)
			mask = mask.to(device,non_blocking=True)
			x0,_ = gpu_norm(vnir,mask)

			t = torch.randint(0,diffusion.timesteps,(x0.shape[0],),device=device,dtype=torch.long)

			# FORWARD -- NOISE x0 TO x_t, PREDICT THE NOISE
			with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):

				noise = torch.randn_like(x0)

				x_t = diffusion.q_sample(x0,t,noise)

				predicted_noise = model(x_t,t).float()

				loss = F.mse_loss(predicted_noise,noise)

			loss_sum  += loss * x0.size(0)
			n_samples += x0.size(0)

	# SYNC
	return {'vloss':(loss_sum/n_samples).item()}


@torch.no_grad()
def sample_and_score(model,encoder,diffusion,real,device,n_samples=2048,batch=64,img_shape=(4,256,256),seed=0):
	'''
	Score n_samples generated images against the validation statistics 'real' (validation_set_statistics()):
	  fid            FID on pooled encoder features (biased by n_samples; keep it fixed)
	  kid, kid_std   KID mean/std over subsets (unbiased, the main feature-based score)
	  w1_<b>         per-band Wasserstein-1 of pixel values, in DN (radiometry/brightness drift)
	  hf_<b>         per-band mean |gradient| generated/real (< 1: smoother/blurrier than real)
	  lpips          mean BT-feature LPIPS between pairs of generated images (n_samples/2 pairs, all bands)
	  lpips_ratio    lpips generated/real (< 1: less diverse than real, e.g. mode collapse)
	The encoder (Barlow Twins) is invariant to blur and gain, so w1_/hf_ cover what fid/kid cannot see.
	The starting noise comes from a generator seeded with 'seed', so every call samples from the
	same x_T and scores are comparable across epochs. The global RNG (training stream) is untouched.
	'''
	assert n_samples % batch == 0, f"n_samples {n_samples} not a multiple of batch {batch}" # FIXED BATCH SHAPE, NO RECOMPILE
	assert batch % 2 == 0, f"batch {batch} must be even: LPIPS pairs the first and second half of each batch"
	model.eval()
	encoder.eval()

	generator = torch.Generator(device=device).manual_seed(seed)
	mean      = dataloader.MEAN.to(device)
	stdd      = dataloader.STDD.to(device)

	feats    = []
	hist     = torch.zeros(img_shape[0],256,dtype=torch.float64,device=device)
	grad_sum = torch.zeros(img_shape[0],dtype=torch.float64,device=device)
	n_diff   = 0
	dists    = []

	# SAMPLE A PREDEFINED NUMBER OF IMAGES, ONE BATCH AT A TIME
	for _ in range(n_samples // batch):

		# STARTING NOISE x_T
		x_t = torch.randn((batch,*img_shape),generator=generator,device=device)

		# PRODUCE IMAGES (NORMALIZED BANDS, CLIPPED TO THE DATA RANGE) AND THEIR FEATURES
		with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=True):
			x0_pred = diffusion.ddim_sample_loop(x_t)              #[B,4,H,W]
			skips,features = encoder(x0_pred)                      #_,[B,C,16,16] -- ALREADY NORMALIZED, NO gpu_norm
			vector = features.mean(dim=(2,3)).double()             #[B,C] -- SAME AS validation_set_statistics()
		feats.append(vector)

		# PIXEL STATISTICS IN DN, SAME AS validation_set_statistics()
		dn = x0_pred.float() * stdd + mean
		for b in range(dn.shape[1]):
			hist[b] += torch.histc(dn[:,b],bins=256,min=-0.5,max=255.5).double()
		grad_sum += (dn[:,:,1:,:]-dn[:,:,:-1,:]).abs().sum(dim=(0,2,3)).double() # VERTICAL
		grad_sum += (dn[:,:,:,1:]-dn[:,:,:,:-1]).abs().sum(dim=(0,2,3)).double() # HORIZONTAL
		n_diff   += dn.shape[0] * ((dn.shape[2]-1)*dn.shape[3] + dn.shape[2]*(dn.shape[3]-1))

		# LPIPS PAIRS: SAMPLES ARE INDEPENDENT, SO PAIR THE FIRST HALF OF THE BATCH WITH THE SECOND
		dists.append(calculate_lpips(encoder,x0_pred[:batch//2],x0_pred[batch//2:]))

	feats = torch.cat(feats) #[n_samples,C]

	# FID
	mu_g    = feats.mean(dim=0)
	sigma_g = torch.cov(feats.T)
	fid     = calculate_fid(real['mu'],real['sigma'],mu_g,sigma_g)

	# KID
	kid,kid_std = calculate_kid(real['feats'],feats)

	# PER-BAND PIXEL DISTRIBUTION AND HIGH-FREQUENCY CONTENT
	w1 = calculate_band_wasserstein(real['hist'],hist)
	hf = (grad_sum/n_diff / real['grad']).tolist() #GPU SYNC

	# INTRA-SET LPIPS (DIVERSITY), RELATIVE TO THE VALIDATION REFERENCE
	lpips_g = torch.cat(dists).mean().item()

	# INSERTION ORDER = COLUMN ORDER: DiffLogger WRITES VALUES IN DICT ORDER UNDER THE SCORE_KEYS HEADER
	scores = {'fid':fid,'kid':kid,'kid_std':kid_std}
	scores.update({f'w1_{b}':w1[b] for b in range(len(w1))})
	scores.update({f'hf_{b}':hf[b] for b in range(len(hf))})
	scores.update({'lpips':lpips_g,'lpips_ratio':lpips_g/real['lpips']})
	assert tuple(scores) == SCORE_KEYS, f"score order {tuple(scores)} != SCORE_KEYS"
	return scores


def train_and_validate(model,encoder,gpu_aug,gpu_norm,diffusion,dataloaders,optimizer,scheduler,device,epochs=100):
	'''
	Train, validate (MSE), score every 10 epochs, log, and keep the best checkpoint by KID
	(only scored epochs after warmup can be saved). KID is unbiased in the sample count, FID is not.
	'''
	logger  = utils.DiffLogger(f'{LOG_DIR}/diff_{model.model_id:03}.tsv')
	tracker = utils.RecentBestTracker(n=1)
	best_kid   = float('inf')
	best_epoch = 0

	# 0.PRECALCULATE VALIDATION REFERENCE STATISTICS (FEATURES, HISTOGRAMS, GRADIENTS, LPIPS)
	real = validation_set_statistics(encoder,dataloaders['validation'],gpu_norm,device)
	print(f"Validation reference LPIPS (diversity): {real['lpips']:.5f}")

	total_time_t0 = time.perf_counter()

	for epoch in range(epochs):

		print(f'--- EPOCH {epoch:04} ---')
		epoch_time_t0 = time.perf_counter()

		# 1.TRAIN
		tr_result = train(model,gpu_aug,diffusion,dataloaders['training'],optimizer,scheduler,device)

		# 2.VALIDATE (MSE ONLY)
		va_result = validate(model,gpu_norm,diffusion,dataloaders['validation'],device)

		# 3.SCORE SAMPLES (FID, KID, W1, HF, LPIPS) -- EVERY 10 EPOCHS
		sample_flag = (epoch+1) % 10 == 0

		if sample_flag:
			scores = sample_and_score(model,encoder,diffusion,real,device)
		else:
			# dummy in logs -- SAME KEYS EVERY EPOCH: DiffLogger FIXES ITS HEADER ON THE FIRST ROW
			scores = {k:float('nan') for k in SCORE_KEYS}

		epoch_time_t1 = time.perf_counter()

		# LOG
		logger.log({'epoch':epoch,**tr_result,**va_result,**scores})
		print(f"[T] LOSS: {tr_result['tloss']:.5f} | [V] LOSS: {va_result['vloss']:.5f}")

		# CHECKPOINTS -- SKIP WARMUP; LOWEST KID SELECTS THE BEST (ONLY SCORED EPOCHS HAVE ONE)
		if sample_flag and (epoch+1)>10 and scores['kid'] < best_kid:
			best_kid   = scores['kid']
			best_epoch = epoch
			pth_path = utils.save_checkpoint(MODEL_DIR,model,optimizer,epoch,tag='diff')
			tracker.update(pth_path)

		# STDOUT LOG EPOCH TIME
		print(f"Epoch time: {epoch_time_t1-epoch_time_t0:.2f} secs.")

	total_time_t1 = time.perf_counter()
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

	# STDOUT LOG OVERALL
	print(f'Total time: {(total_time_t1-total_time_t0)/60:.1f} min')
	print(f'Best epoch: {best_epoch} | KID: {best_kid:.5f}')
	print(f'Peak GPU memory allocated: {mem_gb:.2f} GB')


@torch.no_grad()
def plot_samples(model,diffusion,device):
	'''
	Plot some images generated from random noise. TODO
	'''
	pass


################################################################################
# MAIN/SETUP
################################################################################
if __name__ == '__main__':

	# PARSE ARGS/FILES, SET SEED
	args = parse_args()
	hp   = utils.load_hyperparameters(args)
	utils.set_seed(hp['seed'])
	device = torch.device(f'cuda:{args.device}')

	# CHECK HYPERPARAMETERS
	lrate,decay,epochs,batch = hp['lrate'],hp['decay'],hp['epochs'],hp['batch']
	assert hp['arch'] in ['cnn','vit'], f"Unknown arch {hp['arch']}"
	assert lrate > 0 and decay >= 0, f"Got lrate {lrate}, decay {decay}"
	assert epochs > 10, f"Got {epochs} epochs; 10 are warmup"
	assert (hp['vit_layers'] >= 1) if hp['arch'] == 'vit' else (hp['vit_layers'] == 0), f"Got vit_layers {hp['vit_layers']} for arch {hp['arch']}"

	# THE UNGUIDED DENOISER -- PREDICTS THE NOISE ON ALL 4 BANDS
	ddpm = models.UnconditionedDenoiser(
		hp['id'],
		arch=hp['arch'],
		cnn_layers=hp['cnn_layers'],
		vit_layers=max(hp['vit_layers'],1), # IGNORED BY cnn
		init_chan=hp['init_chan'],
		mlp_ratio=hp.get('mlp_ratio',4),
		time_dim=hp.get('time_dim',256),
		in_channels=4,
		out_channels=4
	)
	ddpm.to(device)
	print(f"{hp['arch']} | init_chan {hp['init_chan']} | cnn_layers {hp['cnn_layers']} | vit_layers {hp['vit_layers']} | batch {batch} | lrate {lrate} | decay {decay}")
	print(f'Parameters: {models.get_model_parameter_size(ddpm):,}')

	# NO DECAY ON NORM WEIGHTS AND BIASES (ndim < 2)
	params = list(ddpm.parameters())
	param_groups = [
		{'params':[p for p in params if p.ndim >= 2],'weight_decay':decay},
		{'params':[p for p in params if p.ndim < 2],'weight_decay':0.0}
	]
	optimizer = torch.optim.AdamW(param_groups,lr=lrate,fused=device.type == 'cuda') # single fused kernel for the whole update

	# AN AUXILIARY FIXED ENCODER TO PRODUCE AN UNBIASED SET OF FEATURES THROUGHOUT TRAINING 
	encoder = load_encoder_checkpoint(hp.get('bt_id'),hp.get('bt_run','bt_pretrain_final'))
	encoder = encoder.to(device)
	encoder = torch.compile(encoder)

	# SCHEDULER -- 10-EPOCH LINEAR WARMUP + COSINE ANNEALING W/O RESTARTS, FLOOR AT lrate/cosine_steps
	warmup_steps = 10
	cosine_steps = epochs-warmup_steps
	min_factor = 1.0 / cosine_steps
	cosine_factor = lambda e: min_factor + (1.0-min_factor) * 0.5 * (1.0 + math.cos(math.pi * min(e,cosine_steps) / cosine_steps))
	warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-2,end_factor=1.0,total_iters=warmup_steps)
	cosine_sched = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=cosine_factor)
	scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_sched,cosine_sched],milestones=[warmup_steps])

	# COMPILE AFTER BUILDING THE OPTIMIZER; save_checkpoint UNWRAPS _orig_mod
	ddpm = torch.compile(ddpm)

	# LOAD DATASET TO RAM
	bands_tr,masks_tr = dataloader.load_cached_split(args.data_dir,split='training')
	bands_va,masks_va = dataloader.load_cached_split(args.data_dir,split='validation')

	# GEOMETRIC AUGS ONLY: GAIN WOULD SHIFT THE BAND DISTRIBUTION THE MODEL IS LEARNING
	gpu_aug  = dataloader.GPULabeledTransform(gain=0.0).to(device)
	gpu_norm = dataloader.GPUValidationTransform().to(device)

	dloader_tr = DataLoader(
		dataset=dataloader.SentinelFastDataset(bands_tr,masks_tr),
		shuffle=True,
		drop_last=True,
		batch_size=batch,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6,
		persistent_workers=True
	)

	dloader_va = DataLoader(
		dataset=dataloader.SentinelFastDataset(bands_va,masks_va),
		shuffle=False,
		drop_last=False,
		batch_size=batch,
		num_workers=args.num_workers,
		pin_memory=True,
		prefetch_factor=6,
		persistent_workers=True
	)
	dataloaders = {'training':dloader_tr,'validation':dloader_va}

	# DIFFUSION PROCESS AROUND THE (COMPILED) DENOISER
	# SAMPLED x0 CLIPPED PER BAND TO THE NORMALIZED uint8 RANGE
	x0_min = ((0.0-dataloader.MEAN)/dataloader.STDD).flatten()
	x0_max = ((255.0-dataloader.MEAN)/dataloader.STDD).flatten()
	diffusion = GaussianDiffusion(ddpm,timesteps=1000,ddim_steps=hp.get('ddim_steps',100),device=device,guided=False,x0_min=x0_min,x0_max=x0_max)

	#(model,encoder,gpu_aug,gpu_norm,diffusion,dataloaders,optimizer,scheduler,device,epochs=100):
	train_and_validate(ddpm,encoder,gpu_aug,gpu_norm,diffusion,dataloaders,optimizer,scheduler,device,epochs=epochs)
