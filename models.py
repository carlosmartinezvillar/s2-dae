import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.flop_counter import FlopCounterMode


################################################################################
# CNN/ViT Blocks
################################################################################
class ConvBlock(nn.Module):
	'''
	Base convolutional block.
	Channel dimension consistent throughout block to match skip/residual.
	'''
	def __init__(self,channels,depth=2):
		super().__init__()
		self.block = nn.ModuleList()
		for _ in range(depth):
			self.block.append(nn.Sequential(
				nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True),
				nn.GroupNorm(1,channels),
				nn.GELU()
			))

	def forward(self,x):
		out = x
		for layer in self.block:
			out = layer(out)
		return x + out


class MultiHeadSelfAttention(nn.Module):
	'''
	Multi-head Self-Attention Operation
	B: batch dimension
	E: embedding dimensino
	N: sequence length
	H: head dimension
	'''
	def __init__(self, E, num_heads=4):
		super().__init__()
		assert E % num_heads == 0, f"channels={E} not divisible by num_heads={num_heads}"
		self.E         = E
		self.num_heads = num_heads
		self.head_dim  = E // num_heads
		self.W_qkv  = nn.Linear(E, E * 3, bias=False)
		self.W_o    = nn.Linear(E, E, bias=False)

	def forward(self, x):
		B, N, _ = x.shape
		QKV = self.W_qkv(x)         # [B,N,3E]
		Q,K,V = QKV.chunk(3,dim=-1) # each [B,N,E]
		Q = Q.view(B,N,self.num_heads,self.head_dim).transpose(1,2) # [B,num_heads,N,H]
		K = K.view(B,N,self.num_heads,self.head_dim).transpose(1,2)
		V = V.view(B,N,self.num_heads,self.head_dim).transpose(1,2)

		# softmax(QK^T/sqrt(H))V with fused flash/memory-efficient kernels: the [N,N] attention
		# matrix is never stored, so memory grows ~linearly (not quadratically) with N
		x = F.scaled_dot_product_attention(Q,K,V) # [B,num_heads,N,H]

		# PREVIOUS (EXPLICIT) VERSION -- stores the full [B,num_heads,N,N] attention matrix
		# attn = (Q @ K.transpose(-2, -1)) # [B,num_heads,N,N]
		# attn = attn / (self.head_dim ** 0.5)
		# attn = attn.softmax(dim=-1)
		# x = attn @ V # [B,num_heads,N,H]

		x = x.transpose(1, 2).reshape(B,N,self.E) #[B,num_heads,N,H] -> [B,N,num_heads,H] -> [B,N,E]
		return self.W_o(x) # [B,N,E]


class MLP(nn.Module):
	'''
	Vanilla MLP layer in transformer block
	'''
	def __init__(self, dim, mlp_ratio=4):
		super().__init__()
		hidden_dim = dim * mlp_ratio
		self.layers = nn.Sequential(
		    nn.Linear(dim, hidden_dim),
		    nn.GELU(),
		    nn.Linear(hidden_dim, dim)
		)

	def forward(self, x):
		return self.layers(x)


class CrossAttention(nn.Module):
	'''
	Multi-head Cross-Attention: queries come from x, keys/values from a context
	token sequence (RGB condition).
	B: batch dimension
	N: query sequence length (x tokens)
	M: context sequence length
	'''
	def __init__(self, E, context_dim, num_heads=4):
		super().__init__()
		assert E % num_heads == 0, f"channels={E} not divisible by num_heads={num_heads}"
		self.E         = E
		self.num_heads = num_heads
		self.head_dim  = E // num_heads
		self.W_q  = nn.Linear(E, E, bias=False)
		self.W_kv = nn.Linear(context_dim, E * 2, bias=False)
		self.W_o  = nn.Linear(E, E, bias=False)

	def forward(self, x, context):
		B,N,_ = x.shape
		_,M,_ = context.shape
		Q    = self.W_q(x)                    # [B,N,E]
		K,V  = self.W_kv(context).chunk(2,dim=-1) # each [B,M,E]
		Q = Q.view(B,N,self.num_heads,self.head_dim).transpose(1,2) # [B,num_heads,N,H]
		K = K.view(B,M,self.num_heads,self.head_dim).transpose(1,2)
		V = V.view(B,M,self.num_heads,self.head_dim).transpose(1,2)

		# softmax(QK^T/sqrt(H))V without storing the [N,M] attention matrix
		x = F.scaled_dot_product_attention(Q,K,V) # [B,num_heads,N,H]

		# PREVIOUS (EXPLICIT) VERSION -- stores the full [B,num_heads,N,M] attention matrix
		# attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B,num_heads,N,M]
		# attn = attn.softmax(dim=-1)
		# x = attn @ V # [B,num_heads,N,H]

		x = x.transpose(1, 2).reshape(B,N,self.E)
		return self.W_o(x)


class ViTLayer(nn.Module):
	'''
	A complete ViT layer (i.e. MHSA + MLP)
	'''
	def __init__(self,E,num_heads,mlp_ratio=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(E)
		self.attn  = MultiHeadSelfAttention(E,num_heads)
		self.norm2 = nn.LayerNorm(E)
		self.mlp   = MLP(E, mlp_ratio)

	def forward(self, tokens):
		tokens = tokens + self.attn(self.norm1(tokens))
		tokens = tokens + self.mlp(self.norm2(tokens))
		return tokens


class ViTBlock(nn.Module):
	'''
	Wrapper for ViT layers for image-token-image conversion.
	Takes inputs [B,C,H,W]. Returns [B,C,H,W]
	'''
	def __init__(self,E,num_heads,mlp_ratio=4,depth=1):
		super().__init__()
		self.block = nn.ModuleList([ViTLayer(E,num_heads,mlp_ratio) for _ in range(depth)])
		# self.block = ViTLayer(E,num_heads,mlp_ratio) #single layer for now

	def forward(self,x):
		B,C,H,W = x.shape
		tokens = x.permute(0,2,3,1).reshape(B,H*W,C)
		# tokens = self.block(tokens)
		for layer in self.block:
			tokens = layer(tokens)
		return tokens.reshape(B,H,W,C).permute(0,3,1,2)


################################################################################
# TIME ADJUSTED LAYERS
################################################################################
class TimeConvBlock(nn.Module):
	'''
	ConvBlock conditioned on embeddings of timestep.
	'''
	def __init__(self,channels,time_dim,depth=2):
		super().__init__()
		self.convs = nn.ModuleList()
		self.norms = nn.ModuleList()
		self.activ = nn.ModuleList()
		self.time_proj = nn.ModuleList()
		for _ in range(depth):
			self.convs.append(nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True))
			self.norms.append(nn.GroupNorm(1,channels))
			self.activ.append(nn.GELU())
			self.time_proj.append(nn.Linear(time_dim,channels*2))

	def forward(self,x,t_emb):
		out = x
		for conv,norm,proj,act in zip(self.convs,self.norms,self.time_proj,self.activ):
			out = conv(out)
			out = norm(out)
			scale,shift = proj(t_emb).chunk(2,dim=-1)
			out = out * (1 + scale[:,:,None,None]) + shift[:,:,None,None]
			out = act(out)
		return x + out


class TimeViTLayer(nn.Module):
	'''
	A complete ViT layer (i.e. MHSA + MLP) conditioned on time embeddings.
	'''
	def __init__(self,E,num_heads,time_dim,mlp_ratio=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(E)
		self.attn  = MultiHeadSelfAttention(E,num_heads)
		self.time_proj1 = nn.Linear(time_dim,E*2)
		self.norm2 = nn.LayerNorm(E)
		self.mlp   = MLP(E, mlp_ratio)
		self.time_proj2 = nn.Linear(time_dim,E*2)

	def forward(self,tokens,t_emb):
		scale,shift = self.time_proj1(t_emb).chunk(2,dim=-1)
		h = self.norm1(tokens) * (1 + scale[:,None,:]) + shift[:,None,:]
		tokens = tokens + self.attn(h)

		scale,shift = self.time_proj2(t_emb).chunk(2,dim=-1)
		h = self.norm2(tokens) * (1 + scale[:,None,:]) + shift[:,None,:]
		tokens = tokens + self.mlp(h)
		return tokens	


class TimeViTBlock(nn.Module):
	'''
	Wrapper for ViT layers for image-token-image conversion.
	'Block' means a grouping intended as equivalent to 'convolutional' block in CNNs.
	Takes an 'image-shaped' feature map [B,C,H,W]. Returns tensor of same shape.
	'''
	def __init__(self,E,num_heads,time_dim,mlp_ratio=4,depth=2):
		super().__init__()
		self.block = nn.ModuleList([TimeViTLayer(E,num_heads,time_dim,mlp_ratio) for _ in range(depth)])

	def forward(self,x,t_emb):
		B,C,H,W = x.shape
		tokens = x.permute(0,2,3,1).reshape(B,H*W,C)
		# tokens = self.block(tokens)
		for layer in self.block:
			tokens = layer(tokens,t_emb)
		return tokens.reshape(B,H,W,C).permute(0,3,1,2)


################################################################################
# BLOCKS COMBINING CONTEXT
################################################################################
class ContextViTLayer(nn.Module):
	'''
	Complete ViT layer with added cross attention for VNIR context
	'''
	def __init__(self,E,num_heads,time_dim,context_dim,mlp_ratio=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(E)
		self.attn  = MultiHeadSelfAttention(E,num_heads)
		self.time_proj1 = nn.Linear(time_dim,E*2)

		self.norm_context = nn.LayerNorm(E)
		self.cross_attn   = CrossAttention(E,context_dim,num_heads)
		# ZERO-INIT OUTPUT: NO CONTEXT EFFECT AT START, SO A TimeViTLayer CHECKPOINT
		# LOADED HERE COMPUTES EXACTLY WHAT IT DID (SAME AS ContextCNNBlock.context_proj)
		nn.init.zeros_(self.cross_attn.W_o.weight)

		self.norm2 = nn.LayerNorm(E)
		self.mlp   = MLP(E, mlp_ratio)
		self.time_proj2 = nn.Linear(time_dim,E*2)

	def forward(self,tokens,t_emb,context):
		scale,shift = self.time_proj1(t_emb).chunk(2,dim=-1)
		h = self.norm1(tokens) * (1 + scale[:,None,:]) + shift[:,None,:]
		tokens = tokens + self.attn(h)

		tokens = tokens + self.cross_attn(self.norm_context(tokens),context)

		scale,shift = self.time_proj2(t_emb).chunk(2,dim=-1)
		h = self.norm2(tokens) * (1 + scale[:,None,:]) + shift[:,None,:]
		tokens = tokens + self.mlp(h)
		return tokens


class ContextViTBlock(nn.Module):
	'''
	Wrapper for ViT layers for image-token-image conversion.
	'Block' means a grouping intended as equivalent to 'convolutional' block in CNNs.
	Takes an 'image-shaped' feature map [B,C,H,W]. Returns tensor of same shape.
	Cross-attends to context, an 'image-shaped' feature map [B,C_context,H,W] with the same
	H,W as x.
	'''
	def __init__(self,E,num_heads,time_dim,context_dim,mlp_ratio=4,depth=1):
		super().__init__()
		self.block = nn.ModuleList([ContextViTLayer(E,num_heads,time_dim,context_dim,mlp_ratio) for _ in range(depth)])
		# self.s_attn = 
		# self.c_attn = ConditionedViTLayer(E,num_heads,time_dim,context_dim,mlp_ratio)

	def forward(self,x,t_emb,context):
		B,C,H,W = x.shape
		tokens = x.permute(0,2,3,1).reshape(B,H*W,C)

		Bc,Cc,Hc,Wc = context.shape
		context_tokens = context.permute(0,2,3,1).reshape(Bc,Hc*Wc,Cc)

		for layer in self.block:
			tokens = layer(tokens,t_emb,context_tokens)
		return tokens.reshape(B,H,W,C).permute(0,3,1,2)


class ContextCNNBlock(nn.Module):
	'''
	ConvBlock conditioned on timestep embeddings and context feature maps.
	Context [B,C_context,H,W] must match the block's H,W; a 1x1 conv projects it to
	the block's channels and it is added at the input of the residual branch, so the
	convs mix it with x and each GroupNorm keeps the sum's scale in check.
	Layer names match TimeConvBlock and context_proj starts at zero, so a TimeConvBlock
	checkpoint loads with strict=False (missing: context_proj only) and the block starts
	out computing exactly what TimeConvBlock did.
	'''
	def __init__(self,channels,time_dim,context_dim,depth=2):
		super().__init__()

		self.convs = nn.ModuleList()
		self.norms = nn.ModuleList()
		self.activ = nn.ModuleList()
		self.time_proj = nn.ModuleList()
		for _ in range(depth):
			self.convs.append(nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=True))
			self.norms.append(nn.GroupNorm(1,channels))
			self.activ.append(nn.GELU())
			self.time_proj.append(nn.Linear(time_dim,channels*2))

		# if op=cross_attention add context by cross-attention, -- TODO
		# else op=simple, simply add the feature maps

		# CONTEXT CHANNELS -> BLOCK CHANNELS. ZERO-INIT: NO CONTEXT EFFECT AT START
		self.context_proj = nn.Conv2d(context_dim,channels,kernel_size=1,bias=True)
		nn.init.zeros_(self.context_proj.weight)
		nn.init.zeros_(self.context_proj.bias)

	def forward(self,x,t_emb,context):
		out = x + self.context_proj(context) # ADD CONTEXT AT THE RESIDUAL BRANCH INPUT
		for conv,norm,proj,act in zip(self.convs,self.norms,self.time_proj,self.activ):
			out = conv(out)
			out = norm(out)
			scale,shift = proj(t_emb).chunk(2,dim=-1)
			out = out * (1 + scale[:,:,None,None]) + shift[:,:,None,None]
			out = act(out)
		return x + out


################################################################################
# DENOISING ENCODERS -- TODO
################################################################################
class UnguidedDenoisingCNNEncoder(nn.Module):
	'''
	5-stage CNN, all blocks timestep-conditioned, no context.
	Takes x already projected to 'channels' (the denoiser's in_layer does that).
	Stage names match the guided encoders, so its weights load into a guided
	encoder with ContextCNNBlock stages (missing: context_proj only).
	'''
	def __init__(self,cnn_layers=3,channels=32,time_dim=256):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		# [H,W]
		self.encoder_1 = TimeConvBlock(channels,time_dim,depth=cnn_layers)
		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		# [H/2,W/2]
		self.encoder_2 = TimeConvBlock(channels*2,time_dim,depth=cnn_layers)
		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)
		# [H/4,W/4]
		self.encoder_3 = TimeConvBlock(channels*4,time_dim,depth=cnn_layers)
		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)
		# [H/8,W/8]
		self.encoder_4 = TimeConvBlock(channels*8,time_dim,depth=cnn_layers)
		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)
		# [H/16,W/16]
		self.encoder_5 = TimeConvBlock(channels*16,time_dim,depth=cnn_layers)

	def forward(self,x,t_emb):
		enc_1 = self.encoder_1(x,t_emb)
		enc_2 = self.encoder_2(self.down_1(enc_1),t_emb)
		enc_3 = self.encoder_3(self.down_2(enc_2),t_emb)
		enc_4 = self.encoder_4(self.down_3(enc_3),t_emb)
		enc_5 = self.encoder_5(self.down_4(enc_4),t_emb)
		return [enc_1,enc_2,enc_3,enc_4], enc_5


class UnguidedDenoisingViTEncoder(nn.Module):
	'''
	Hybrid CNN/ViT without positional encoding. 2xCNN + 3xViT stages.
	All blocks timestep-conditioned, no context.
	Takes x already projected to 'channels' (the denoiser's in_layer does that).
	Stage names, heads and depths match GuidedDenoisingViTEncoder, so its weights load
	there (missing: cross_attn and norm_context only).
	'''
	def __init__(self,cnn_layers=3,vit_layers=1,channels=32,mlp_ratio=4,time_dim=256):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		# depth PASSED EXPLICITLY: TimeViTBlock DEFAULTS TO 2, ContextViTBlock TO 1
		self.encoder_1 = TimeConvBlock(channels,time_dim,depth=cnn_layers)
		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		self.encoder_2 = TimeConvBlock(channels*2,time_dim,depth=cnn_layers)
		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)
		self.encoder_3 = TimeViTBlock(channels*4,num_heads=2,time_dim=time_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)
		self.encoder_4 = TimeViTBlock(channels*8,num_heads=4,time_dim=time_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)
		self.encoder_5 = TimeViTBlock(channels*16,num_heads=8,time_dim=time_dim,mlp_ratio=mlp_ratio,depth=vit_layers)

	def forward(self,x,t_emb):
		enc_1 = self.encoder_1(x,t_emb)
		enc_2 = self.encoder_2(self.down_1(enc_1),t_emb)
		enc_3 = self.encoder_3(self.down_2(enc_2),t_emb)
		enc_4 = self.encoder_4(self.down_3(enc_3),t_emb)
		enc_5 = self.encoder_5(self.down_4(enc_4),t_emb)
		return [enc_1,enc_2,enc_3,enc_4], enc_5


class GuidedDenoisingCNNEncoder(nn.Module):
	'''
	5-stage CNN, all blocks timestep-conditioned, context added at every stage.
	context_dims: BT encoder channels at c1..c5 (e.g. init_chan*(1,2,4,8,16)).
	Stage names match UnguidedDenoisingCNNEncoder, so its weights load here with
	only the zero-initialized context_proj layers missing.
	'''
	def __init__(self,cnn_layers=3,channels=32,time_dim=256,context_dims=(32,64,128,256,512)):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		c1_dim,c2_dim,c3_dim,c4_dim,c5_dim = context_dims

		# [H,W]
		self.encoder_1 = ContextCNNBlock(channels,time_dim,c1_dim,depth=cnn_layers)
		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		# [H/2,W/2]
		self.encoder_2 = ContextCNNBlock(channels*2,time_dim,c2_dim,depth=cnn_layers)
		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)
		# [H/4,W/4]
		self.encoder_3 = ContextCNNBlock(channels*4,time_dim,c3_dim,depth=cnn_layers)
		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)
		# [H/8,W/8]
		self.encoder_4 = ContextCNNBlock(channels*8,time_dim,c4_dim,depth=cnn_layers)
		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)
		# [H/16,W/16]
		self.encoder_5 = ContextCNNBlock(channels*16,time_dim,c5_dim,depth=cnn_layers)

	def forward(self,x,t_emb,context):
		(c1,c2,c3,c4),c5 = context # BT ENCODER OUTPUT: ([c1,c2,c3,c4],c5)
		enc_1 = self.encoder_1(x,t_emb,c1)
		enc_2 = self.encoder_2(self.down_1(enc_1),t_emb,c2)
		enc_3 = self.encoder_3(self.down_2(enc_2),t_emb,c3)
		enc_4 = self.encoder_4(self.down_3(enc_3),t_emb,c4)
		enc_5 = self.encoder_5(self.down_4(enc_4),t_emb,c5)
		return [enc_1,enc_2,enc_3,enc_4], enc_5

class GuidedDenoisingViTEncoder(nn.Module):
	'''
	Hybrid CNN/ViT without positional encoding. 2xCNN + 3xViT layers.
	All blocks timestep-conditioned.
	'''

	def __init__(self,cnn_layers=3,vit_layers=1,channels=32,mlp_ratio=4,time_dim=256,context_dims=(128,256,512)):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		c3_dim,c4_dim,c5_dim = context_dims

		self.encoder_1 = TimeConvBlock(channels,time_dim,depth=cnn_layers) #32
		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		self.encoder_2 = TimeConvBlock(channels*2,time_dim,depth=cnn_layers)
		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)
		self.encoder_3 = ContextViTBlock(channels*4,num_heads=2,time_dim=time_dim,context_dim=c3_dim,mlp_ratio=mlp_ratio,depth=vit_layers)		
		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)
		self.encoder_4 = ContextViTBlock(channels*8,num_heads=4,time_dim=time_dim,context_dim=c4_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)
		self.encoder_5 = ContextViTBlock(channels*16,num_heads=8,time_dim=time_dim,context_dim=c5_dim,mlp_ratio=mlp_ratio,depth=vit_layers)	

	def forward(self,x,t_emb,context):
		(_,_,c3,c4),c5 = context # BT ENCODER OUTPUT: ([c1,c2,c3,c4],c5)
		enc_1 = self.encoder_1(x,t_emb)
		enc_2 = self.encoder_2(self.down_1(enc_1),t_emb)
		enc_3 = self.encoder_3(self.down_2(enc_2),t_emb,c3)
		enc_4 = self.encoder_4(self.down_3(enc_3),t_emb,c4)
		enc_5 = self.encoder_5(self.down_4(enc_4),t_emb,c5)
		return [enc_1,enc_2,enc_3,enc_4], enc_5


################################################################################
# DENOISING DECODERS -- TODO
################################################################################
class UnguidedDenoisingCNNDecoder(nn.Module):
	'''
	5-stage CNN, all blocks timestep-conditioned, no context. Mirrors UnguidedDenoisingCNNEncoder.
	Returns features at 'channels' width; the denoiser's out_layer maps them to the output.
	Stage names match the guided decoders, so its weights load into a guided decoder
	with ContextCNNBlock stages (missing: context_proj only).
	'''
	def __init__(self,cnn_layers=3,channels=32,time_dim=256):
		super().__init__()
		up_params = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True}

		# [H/16,W/16]
		self.decoder_1 = TimeConvBlock(channels*16,time_dim,depth=cnn_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)
		# [H/8,W/8]
		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = TimeConvBlock(channels*8,time_dim,depth=cnn_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)
		# [H/4,W/4]
		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = TimeConvBlock(channels*4,time_dim,depth=cnn_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)
		# [H/2,W/2]
		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = TimeConvBlock(channels*2,time_dim,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)
		# [H,W]
		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = TimeConvBlock(channels,time_dim,depth=cnn_layers)

	def forward(self,x,skips,t_emb):
		enc_1,enc_2,enc_3,enc_4 = skips
		dec_1 = self.decoder_1(x,t_emb)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ),t_emb)
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ),t_emb)
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ),t_emb)
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ),t_emb)
		return dec_5


class UnguidedDenoisingViTDecoder(nn.Module):
	'''
	3-stage ViT & 2-stage CNN, all blocks timestep-conditioned, no context.
	Mirrors UnguidedDenoisingViTEncoder. Returns features at 'channels' width; the
	denoiser's out_layer maps them to the output. Stage names, heads and depths match
	GuidedDenoisingViTDecoder, so its weights load there (missing: cross_attn and
	norm_context only).
	'''
	def __init__(self,cnn_layers=3,vit_layers=1,channels=32,mlp_ratio=4,time_dim=256):
		super().__init__()
		up_params = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True}

		# depth PASSED EXPLICITLY: TimeViTBlock DEFAULTS TO 2, ContextViTBlock TO 1
		self.decoder_1 = TimeViTBlock(channels*16,num_heads=8,time_dim=time_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)

		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = TimeViTBlock(channels*8,num_heads=4,time_dim=time_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)

		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = TimeViTBlock(channels*4,num_heads=2,time_dim=time_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)

		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = TimeConvBlock(channels*2,time_dim,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)

		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = TimeConvBlock(channels,time_dim,depth=cnn_layers)

	def forward(self,x,skips,t_emb):
		enc_1,enc_2,enc_3,enc_4 = skips
		dec_1 = self.decoder_1(x,t_emb)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ),t_emb)
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ),t_emb)
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ),t_emb)
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ),t_emb)
		return dec_5


class GuidedDenoisingCNNDecoder(nn.Module):
	'''
	5-stage CNN, all blocks timestep-conditioned, context added at every stage.
	Mirrors GuidedDenoisingCNNEncoder. context_dims: BT encoder channels at c1..c5.
	Returns features at 'channels' width; the denoiser's out_layer maps them to the output.
	Stage names match UnguidedDenoisingCNNDecoder, so its weights load here with
	only the zero-initialized context_proj layers missing.
	'''
	def __init__(self,cnn_layers=3,channels=32,time_dim=256,context_dims=(32,64,128,256,512)):
		super().__init__()
		up_params = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': True}

		c1_dim,c2_dim,c3_dim,c4_dim,c5_dim = context_dims

		# [H/16,W/16]
		self.decoder_1 = ContextCNNBlock(channels*16,time_dim,c5_dim,depth=cnn_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)
		# [H/8,W/8]
		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = ContextCNNBlock(channels*8,time_dim,c4_dim,depth=cnn_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)
		# [H/4,W/4]
		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = ContextCNNBlock(channels*4,time_dim,c3_dim,depth=cnn_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)
		# [H/2,W/2]
		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = ContextCNNBlock(channels*2,time_dim,c2_dim,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)
		# [H,W]
		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = ContextCNNBlock(channels,time_dim,c1_dim,depth=cnn_layers)

	def forward(self,x,skips,t_emb,context):
		enc_1,enc_2,enc_3,enc_4 = skips
		(c1,c2,c3,c4),c5 = context # BT ENCODER OUTPUT: ([c1,c2,c3,c4],c5)
		dec_1 = self.decoder_1(x,t_emb,c5)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ),t_emb,c4)
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ),t_emb,c3)
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ),t_emb,c2)
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ),t_emb,c1)
		return dec_5

class GuidedDenoisingViTDecoder(nn.Module):
	'''
	3-stage ViT & 2-stage CNN. Mirrors ViTEncoder().
	'''
	def __init__(self,cnn_layers=3,vit_layers=1,channels=32,mlp_ratio=4,time_dim=256,context_dims=(128,256,512)):
		super().__init__()	
		up_params = {'kernel_size': 4, 'stride': 2,'padding': 1, 'bias': True}
		c3_dim,c4_dim,c5_dim = context_dims

		self.decoder_1 = ContextViTBlock(channels*16,num_heads=8,time_dim=time_dim,context_dim=c5_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)

		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = ContextViTBlock(channels*8,num_heads=4,time_dim=time_dim,context_dim=c4_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)

		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = ContextViTBlock(channels*4,num_heads=2,time_dim=time_dim,context_dim=c3_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)

		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = TimeConvBlock(channels*2,time_dim,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)

		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = TimeConvBlock(channels,time_dim,depth=cnn_layers)


	def forward(self,x,skips,t_emb,context):
		enc_1,enc_2,enc_3,enc_4 = skips
		(_,_,c3,c4),c5 = context # BT ENCODER OUTPUT: ([c1,c2,c3,c4],c5)
		dec_1 = self.decoder_1(x,t_emb,c5)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ),t_emb,c4)
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ),t_emb,c3)
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ),t_emb)
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ),t_emb)
		return dec_5


################################################################################
# BT CONDITION/ALIGNMENT ENCODERS
###############################################################################
class ConditionCNNEncoder(nn.Module):
	'''
	Encoder for RGB inputs.
	'''
	def __init__(self,channels=32,cnn_layers=2,in_channels=4):
		super().__init__()
		down_config = {'kernel_size': 3, 'stride': 2,'padding': 1, 'bias': True}

		self.in_layer = nn.Conv2d(in_channels,channels,3,1,1,bias=True)

		# [H,W]
		self.stage_1 = ConvBlock(channels,depth=cnn_layers)
		self.down_1  = nn.Conv2d(channels,channels*2,**down_config)
		# [H/2,W/2]
		self.stage_2 = ConvBlock(channels*2,depth=cnn_layers)
		self.down_2  = nn.Conv2d(channels*2,channels*4,**down_config)
		# [H/4,W/4]
		self.stage_3 = ConvBlock(channels*4,depth=cnn_layers)
		self.down_3  = nn.Conv2d(channels*4,channels*8,**down_config)
		# [H/8,W/8]
		self.stage_4 = ConvBlock(channels*8,depth=cnn_layers)
		self.down_4  = nn.Conv2d(channels*8,channels*16,**down_config)
		# [H/16,W/16]
		self.stage_5 = ConvBlock(channels*16,depth=cnn_layers)

	def forward(self,x):
		x = self.in_layer(x)
		c1 = self.stage_1(x)
		c2 = self.stage_2(self.down_1(c1))
		c3 = self.stage_3(self.down_2(c2))
		c4 = self.stage_4(self.down_3(c3))
		c5 = self.stage_5(self.down_4(c4))
		return [c1,c2,c3,c4],c5


class ConditionViTEncoder(nn.Module):
	'''
	Encoder for RGB inputs.
	Hybrid ViT without positional encoding. 2xCNN + 3xViT layers
	'''
	def __init__(self,channels=32,cnn_layers=2,mlp_ratio=4,in_channels=4):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		self.in_layer = nn.Conv2d(in_channels,channels,3,1,1,bias=True)

		self.encoder_1 = ConvBlock(channels,depth=cnn_layers) #32
		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		self.encoder_2 = ConvBlock(channels*2,depth=cnn_layers)
		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)
		self.encoder_3 = ViTBlock(channels*4,num_heads=2,mlp_ratio=mlp_ratio,depth=1)		
		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)
		self.encoder_4 = ViTBlock(channels*8,num_heads=4,mlp_ratio=mlp_ratio,depth=1)
		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)
		self.encoder_5 = ViTBlock(channels*16,num_heads=8,mlp_ratio=mlp_ratio,depth=1)	

	def forward(self,x):
		x = self.in_layer(x)
		enc_1 = self.encoder_1(x)
		enc_2 = self.encoder_2(self.down_1(enc_1))
		enc_3 = self.encoder_3(self.down_2(enc_2))
		enc_4 = self.encoder_4(self.down_3(enc_3))
		enc_5 = self.encoder_5(self.down_4(enc_4))
		return [enc_1,enc_2,enc_3,enc_4], enc_5


class Projector(nn.Module):
	'''
	Barlow Twins projector. Takes [B,D] inputs.
	'''
	def __init__(self,in_dim,out_dim=4096):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(in_dim, out_dim),
			nn.BatchNorm1d(out_dim),
			nn.ReLU(),
			nn.Linear(out_dim, out_dim) # Final projection dimension
		)
	def forward(self,x):
		return self.layers(x)


class BarlowTwinsLoss(nn.Module):
	'''
	L = sum_i (1 - C_ii)^2 + lambda * sum_i sum_{j!=i} C_ij^2
	C is the cross-correlation matrix between the batch-standardized embeddings
	of the two views. Computed in float32 regardless of autocast.
	'''
	def __init__(self,proj_dim,lambd=5e-3):
		super().__init__()
		self.lambd = lambd
		self.bn    = nn.BatchNorm1d(proj_dim,affine=False,track_running_stats=False)

	def forward(self,z1,z2):

		# DO IN FLOAT 32
		with torch.autocast(device_type='cuda' if z1.is_cuda else 'cpu',enabled=False):

			# NORMALIZE ON BATCH MEAN AND STD DEV.
			z1 = self.bn(z1.float())
			z2 = self.bn(z2.float())

			# CALCULATE CORRELATION MATRIX
			c  = z1.T @ z2 / z1.shape[0] # [D,D]

			diag     = torch.diagonal(c)
			on_diag  = (diag - 1).pow(2).sum()
			off_diag = c.pow(2).sum() - diag.pow(2).sum()
			loss     = on_diag + self.lambd * off_diag

		return loss, on_diag.detach(), off_diag.detach()


class BarlowTwins(nn.Module):
	def __init__(self,model_id,init_chan=32,cnn_layers=2,proj_dim=8192,encoder='cnn'):
		super().__init__()

		# META
		self.model_name = "S2BT"
		self.model_id = model_id

		# ENCODER
		assert encoder in ["cnn","vit"]
		if encoder == 'cnn':
			self.encoder = ConditionCNNEncoder(channels=init_chan,cnn_layers=cnn_layers)
		if encoder == 'vit':
			self.encoder = ConditionViTEncoder(channels=init_chan,cnn_layers=cnn_layers,mlp_ratio=4)

		# PROJECTOR
		self.rep_dim = init_chan*16 #5 stages, 4 downsamplings doubling C -> c5 has init_chan*16
		self.projector = Projector(self.rep_dim,proj_dim)

	def forward(self,x1,x2):
		skips1,rep1_b5 = self.encoder(x1) # ([c1,c2,c3,c4],c5)

		if x2 is None:
			return skips1,rep1_b5

		_,rep2_b5 = self.encoder(x2)

		# Global Average Pooling: [B,C,H,W] -> [B,C]
		emb1 = self.projector(rep1_b5.mean(dim=(2,3)))
		emb2 = self.projector(rep2_b5.mean(dim=(2,3)))

		return emb1,emb2


################################################################################
# BASE/STANDARD MODEL
################################################################################
class BaseCNNSegDecoder(nn.Module):
	'''
	5-stage CNN.
	'''
	def __init__(self,channels=32,cnn_layers=2):
		super().__init__()
		up_params = {'kernel_size': 4, 'stride': 2,'padding': 1, 'bias': True}

		self.decoder_1 = ConvBlock(channels*16,depth=cnn_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)

		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = ConvBlock(channels*8,depth=cnn_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)

		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = ConvBlock(channels*4,depth=cnn_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)

		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = ConvBlock(channels*2,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)

		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = ConvBlock(channels,depth=cnn_layers)

		self.out_layer = nn.Conv2d(channels,2,kernel_size=1,padding=0)

	def forward(self,x,skips):
		enc_1,enc_2,enc_3,enc_4 = skips
		dec_1 = self.decoder_1(x)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ))
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ))
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ))
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ))
		return self.out_layer(dec_5)


class BaseViTSegDecoder(nn.Module):
	'''
	3-stage ViT & 2-stage CNN.
	'''
	def __init__(self,channels=32,cnn_layers=2,vit_layers=1,mlp_ratio=4):
		super().__init__()	
		up_params = {'kernel_size': 4, 'stride': 2,'padding': 1, 'bias': True}

		self.decoder_1 = ViTBlock(channels*16,num_heads=8,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)

		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = ViTBlock(channels*8,num_heads=4,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)

		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = ViTBlock(channels*4,num_heads=2,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)

		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = ConvBlock(channels*2,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)

		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = ConvBlock(channels,depth=cnn_layers)

		self.out_layer = nn.Conv2d(channels,2,kernel_size=1,padding=0)


	def forward(self,x,skips):
		enc_1,enc_2,enc_3,enc_4 = skips
		dec_1 = self.decoder_1(x)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ))
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ))
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ))
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ))
		return self.out_layer(dec_5)


################################################################################
# Time Embeddings
################################################################################
class SinusoidalTimeEmbedding(nn.Module):
	'''
	Standard transformer/DDPM sinusoidal timestep embedding: t -> [B,dim]
	'''
	def __init__(self,dim):
		super().__init__()
		self.dim = dim

	def forward(self,t):
		half  = self.dim // 2
		freqs = torch.exp(-math.log(10000) * torch.arange(half,device=t.device).float() / (half - 1))
		args  = t.float()[:,None] * freqs[None,:]
		emb   = torch.cat([torch.sin(args),torch.cos(args)],dim=-1)
		if self.dim % 2 == 1:
			emb = torch.cat([emb,torch.zeros_like(emb[:,:1])],dim=-1)
		return emb


class TimeEmbedding(nn.Module):
	'''
	Sinusoidal embedding + MLP, projecting timestep t to time_dim. Shared and
	passed into every AdaGN/AdaLN-modulated block in the encoder/decoder.
	'''
	def __init__(self,dim,time_dim):
		super().__init__()
		self.pos_emb = SinusoidalTimeEmbedding(dim)
		self.mlp = nn.Sequential(
			nn.Linear(dim,time_dim),
			nn.GELU(),
			nn.Linear(time_dim,time_dim)
		)

	def forward(self,t):
		return self.mlp(self.pos_emb(t))

################################################################################
# DIFFUSION
################################################################################
class UnconditionedDenoiser(nn.Module):
	'''
	Unguided time-conditioned U-Net on the 4 VNIR bands. arch picks the denoising
	encoder/decoder: 'cnn' (5 CNN stages) or 'vit' (2 CNN + 3 ViT stages).
	- diff_pretrain: out_channels=4, predicts the noise added to the bands at step t.
	- diff_train:    out_channels=2, segmentation logits from the bands at a fixed t.
	A pretrained checkpoint loads into the diff_train model with only out_layer missing.
	'''
	def __init__(self,model_id,arch='vit',cnn_layers=3,vit_layers=1,init_chan=16,mlp_ratio=4,time_dim=256,in_channels=4,out_channels=4):
		super().__init__()

		# META
		self.model_name = "S2Diff"
		self.model_id   = model_id

		# time embeddings
		self.time_embedder = TimeEmbedding(init_chan,time_dim=time_dim)

		# Denoiser input projection layer to 'channels' argument
		self.in_layer = nn.Conv2d(in_channels,init_chan,kernel_size=1,stride=1,padding=0,bias=True)

		# Denoising encoder & decoder
		assert arch in ["cnn","vit"]
		if arch == 'cnn':
			self.encoder = UnguidedDenoisingCNNEncoder(cnn_layers=cnn_layers,channels=init_chan,time_dim=time_dim)
			self.decoder = UnguidedDenoisingCNNDecoder(cnn_layers=cnn_layers,channels=init_chan,time_dim=time_dim)
		if arch == 'vit':
			self.encoder = UnguidedDenoisingViTEncoder(cnn_layers=cnn_layers,vit_layers=vit_layers,channels=init_chan,mlp_ratio=mlp_ratio,time_dim=time_dim)
			self.decoder = UnguidedDenoisingViTDecoder(cnn_layers=cnn_layers,vit_layers=vit_layers,channels=init_chan,mlp_ratio=mlp_ratio,time_dim=time_dim)

		# final 1x1 -- predicted noise (diff_pretrain) or class logits (diff_train)
		self.out_layer = nn.Conv2d(init_chan,out_channels,kernel_size=1,bias=True)

	def forward(self,x_t,t):
		t_emb = self.time_embedder(t) # t: [B] long timesteps

		x             = self.in_layer(x_t)
		skips,enc_out = self.encoder(x,t_emb)
		dec_out       = self.decoder(enc_out,skips,t_emb)
		return self.out_layer(dec_out)

################################################################################
# SEGMENTATION MODELS
################################################################################
class ConditionedDenoiser(nn.Module):

	def __init__(self,model_id,cnn_layers=2,vit_layers=1,init_chan_cond=32,init_chan_diff=16,time_dim=256):
		super().__init__()

		self.model_name = "S2SegDiff"
		self.model_id   = model_id

		# time embeddings
		self.time_embedder = TimeEmbedding(init_chan_diff,time_dim=time_dim)

		# context embeddings
		context_dims = (init_chan_cond*2,init_chan_cond*4,init_chan_cond*8,init_chan_cond*16)
		self.cond_encoder  = ConditionCNNEncoder(channels=init_chan_cond,cnn_layers=cnn_layers)

		# Denoiser input projection layer to 'channels' argument
		self.in_layer = nn.Conv2d(2,init_chan_diff,kernel_size=1,stride=1,padding=0,bias=True)

		# Denoising encoder & decoder
		self.encoder = DenoisingCNNEncoder(cnn_layers,vit_layers,channels,mlp_ratio,time_dim,context_dims)
		self.decoder = DenoisingCNNDecoder(cnn_layers,vit_layers,channels,mlp_ratio,time_dim,context_dims)

		# final 1x1 classifier
		self.out_layer = nn.Conv2d(channels,mask_channels,kernel_size=1,bias=True)

	def forward(self,x_t,t,vnir_image):
		t_emb   = self.time_embedder(t)
		context = self.cond_encoder(vnir_image)

		x             = self.in_layer(x_t)
		skips,enc_out = self.encoder(x,t_emb,context)
		dec_out       = self.decoder(enc_out,skips,t_emb,context)
		return self.out_layer(dec_out)


class BTSegmenter(nn.Module):
	def __init__(self,model_id,encoder,cnn_layers=2):
		super().__init__()
		self.model_id = model_id
		self.encoder = encoder
		# DECODER WIDTH FOLLOWS THE ENCODER'S init_chan SO SKIP CHANNELS MATCH
		self.decoder = BaseCNNSegDecoder(channels=encoder.in_layer.out_channels,cnn_layers=cnn_layers)

	def forward(self,x):
		skips,x = self.encoder(x) # skips = [c1,c2,c3,c4], x = c5
		return self.decoder(x,skips)


class BaseSegmenter(nn.Module): # TODO --- ADD CHANGEABLE ENCODER/DECODER
	'''
	Standard segmentation model supervised learning. Baseline.
	'''
	def __init__(self,channels=32,cnn_layers=2):
		super().__init__()
		self.encoder = ConditionCNNEncoder(channels=channels,cnn_layers=cnn_layers)
		# self.encoder = ConditionViTEncoder(channels=channels,cnn_layers=cnn_layers)
		self.decoder = BaseCNNSegDecoder(channels=channels,cnn_layers=cnn_layers)
		# self.decoder = BaseViTSegDecoder(channels=channels,cnn_layers=cnn_layers)

	def forward(self,x):
		skips, enc_out = self.encoder(x)
		dec_out = self.decoder(enc_out,skips)
		return dec_out


################################################################################
# SOME UTILITY FUNCTIONS
################################################################################
def get_model_memory_footprint(model,H=256,W=256,batch_size=1,backward=True,bf16=True):
	'''
	Peak GPU memory (bytes) of one forward pass (and backward, if backward=True) of S2SegDiff
	on [batch_size,C,H,W] inputs, measured with torch.cuda.max_memory_allocated().
	Includes weights, activations and gradients; bf16=True runs under bf16 autocast like train.py.
	Does not include optimizer state (AdamW: 2x the parameter bytes) unless it is already allocated.
	Requires the model on a CUDA device.
	'''
	model  = model._orig_mod if hasattr(model,'_orig_mod') else model # unwrap torch.compile
	device = next(model.parameters()).device
	if device.type != 'cuda':
		raise ValueError(f"get_model_memory_footprint needs the model on a CUDA device, got '{device}'")

	x_t,t,rgb = _dummy_inputs(model,H,W,batch_size)

	torch.cuda.synchronize(device)
	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats(device)

	use_bf16 = bf16 and torch.cuda.is_bf16_supported()
	with torch.set_grad_enabled(backward), torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=use_bf16):
		out = model(x_t,t,rgb)
	if backward:
		out.float().sum().backward()

	torch.cuda.synchronize(device)
	peak = torch.cuda.max_memory_allocated(device)

	if backward:
		model.zero_grad(set_to_none=True)
	return peak


def get_model_parameter_size(model,trainable_only=False):
	'''
	Number of parameters in the model (only those with requires_grad if trainable_only).
	'''
	model = model._orig_mod if hasattr(model,'_orig_mod') else model # unwrap torch.compile
	return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def count_flops(model,H=256,W=256,batch_size=1,backward=False):
	'''
	FLOPs of one forward pass (and backward, if backward=True) of S2SegDiff on
	[batch_size,C,H,W] inputs, counted with torch.utils.flop_counter.FlopCounterMode.
	Input channels are read from the model's input layers. One multiply-add = 2 FLOPs.
	'''
	model     = model._orig_mod if hasattr(model,'_orig_mod') else model # unwrap torch.compile
	x_t,t,rgb = _dummy_inputs(model,H,W,batch_size)

	flop_counter = FlopCounterMode(display=False)
	with torch.set_grad_enabled(backward), flop_counter:
		out = model(x_t,t,rgb)
		if backward:
			out.sum().backward()
	if backward:
		model.zero_grad(set_to_none=True)

	return flop_counter.get_total_flops()


def _dummy_inputs(model,H,W,batch_size):
	'''
	Random (x_t,t,rgb) inputs for S2SegDiff on the model's device.
	Input channels are read from the model's input layers.
	'''
	device        = next(model.parameters()).device
	mask_channels = model.in_layer.in_channels
	in_channels   = model.cond_encoder.in_layer.in_channels
	x_t = torch.randn(batch_size,mask_channels,H,W,device=device)
	t   = torch.randint(0,1000,(batch_size,),device=device)
	rgb = torch.randn(batch_size,in_channels,H,W,device=device)
	return x_t,t,rgb


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	# MODEL SIZE / COST SUMMARY FOR ONE CONFIGURATION
	cfg = {
		'bands':4,
		'labels':2,
		'channels':32,
		'cnn_layers':2,
		'vit_layers':1,
		'mlp_ratio':4,
		'size':256, # chip height/width
		'batch':32
	}

	from diffusion import mask_channels
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model  = S2SegDiff(model_id=0,mask_channels=mask_channels(cfg['labels']),in_channels=cfg['bands'],cnn_layers=cfg['cnn_layers'],
						vit_layers=cfg['vit_layers'],channels=cfg['channels'],mlp_ratio=cfg['mlp_ratio']).to(device)

	n_params = get_model_parameter_size(model)
	fwd      = count_flops(model,H=cfg['size'],W=cfg['size'],batch_size=1)
	fwd_bwd  = count_flops(model,H=cfg['size'],W=cfg['size'],batch_size=1,backward=True)

	print(f"S2SegDiff: bands={cfg['bands']} labels={cfg['labels']} channels={cfg['channels']} cnn_layers={cfg['cnn_layers']} "
		f"vit_layers={cfg['vit_layers']} mlp_ratio={cfg['mlp_ratio']} | input {cfg['size']}x{cfg['size']}")
	print(f"  parameters:          {n_params:,} ({n_params*4/2**20:.1f} MiB in float32)")
	print(f"  FLOPs per chip:      forward {fwd/1e9:.1f} G | forward+backward {fwd_bwd/1e9:.1f} G")
	print(f"  FLOPs per batch {cfg['batch']}: training step {cfg['batch']*fwd_bwd/1e12:.2f} T")

	if device.type == 'cuda':
		peak = get_model_memory_footprint(model,H=cfg['size'],W=cfg['size'],batch_size=cfg['batch'],backward=True)
		print(f"  peak GPU memory, training step at batch {cfg['batch']}: {peak/2**30:.2f} GiB "
			f"(+{2*n_params*4/2**30:.2f} GiB AdamW state)")
	else:
		print("  peak GPU memory:     skipped (no CUDA device)")