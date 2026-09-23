import torch
import torch.nn as nn
import math
from torch.utils.flop_counter import FlopCounterMode


################################################################################
# CNN Blocks
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


class TimeConditionedConvBlock(nn.Module):
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

################################################################################
# ViT Blocks
################################################################################
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

		attn = (Q @ K.transpose(-2, -1)) # [B,num_heads,N,N]
		attn = attn / (self.head_dim ** 0.5)
		attn = attn.softmax(dim=-1)

		x = attn @ V # [B,num_heads,N,H]
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

		attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B,num_heads,N,M]
		attn = attn.softmax(dim=-1)

		x = attn @ V # [B,num_heads,N,H]
		x = x.transpose(1, 2).reshape(B,N,self.E)
		return self.W_o(x)


class ViTLayer(nn.Module):
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


class ConditionedViTLayer(nn.Module):
	'''
	Complete ViT layer with added cross attention for RGB context
	'''
	def __init__(self,E,num_heads,time_dim,context_dim,mlp_ratio=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(E)
		self.attn  = MultiHeadSelfAttention(E,num_heads)
		self.time_proj1 = nn.Linear(time_dim,E*2)

		self.norm_context = nn.LayerNorm(E)
		self.cross_attn   = CrossAttention(E,context_dim,num_heads)

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


class ViTBlock(nn.Module):
	'''
	Wrapper for ViT layers for image-token-image conversion.
	'Block' means a grouping intended as equivalent to 'convolutional' block in CNNs.
	Takes an 'image-shaped' feature map [B,C,H,W]. Returns tensor of same shape.
	'''
	def __init__(self,E,num_heads,time_dim,mlp_ratio=4,depth=2):
		super().__init__()
		self.block = nn.ModuleList([ViTLayer(E,num_heads,time_dim,mlp_ratio) for _ in range(depth)])

	def forward(self,x,t_emb):
		B,C,H,W = x.shape
		tokens = x.permute(0,2,3,1).reshape(B,H*W,C)
		# tokens = self.block(tokens)
		for layer in self.block:
			tokens = layer(tokens,t_emb)
		return tokens.reshape(B,H,W,C).permute(0,3,1,2)


class ConditionedViTBlock(nn.Module):
	'''
	Wrapper for ViT layers for image-token-image conversion.
	'Block' means a grouping intended as equivalent to 'convolutional' block in CNNs.
	Takes an 'image-shaped' feature map [B,C,H,W]. Returns tensor of same shape.
	Cross-attends to context, an 'image-shaped' feature map [B,C_context,H,W] with the same
	H,W as x.
	'''
	def __init__(self,E,num_heads,time_dim,context_dim,mlp_ratio=4,depth=1):
		super().__init__()
		self.block = nn.ModuleList([ConditionedViTLayer(E,num_heads,time_dim,context_dim,mlp_ratio) for _ in range(depth)])
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


################################################################################
# Encoder(s)
################################################################################
class ViTEncoder(nn.Module):
	'''
	Hybrid CNN/ViT without positional encoding. 2xCNN + 3xViT layers.
	All blocks timestep-conditioned.
	'''

	def __init__(self,cnn_layers=3,vit_layers=1,channels=32,mlp_ratio=4,time_dim=256,context_dims=(128,256,512)):
		super().__init__()
		down_params = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'bias': True}

		c3_dim,c4_dim,c5_dim = context_dims

		self.encoder_1 = TimeConditionedConvBlock(channels,time_dim,depth=cnn_layers) #32
		self.down_1    = nn.Conv2d(channels,channels*2,**down_params)
		self.encoder_2 = TimeConditionedConvBlock(channels*2,time_dim,depth=cnn_layers)
		self.down_2    = nn.Conv2d(channels*2,channels*4,**down_params)
		self.encoder_3 = ConditionedViTBlock(channels*4,num_heads=2,time_dim=time_dim,context_dim=c3_dim,mlp_ratio=mlp_ratio,depth=vit_layers)		
		self.down_3    = nn.Conv2d(channels*4,channels*8,**down_params)
		self.encoder_4 = ConditionedViTBlock(channels*8,num_heads=4,time_dim=time_dim,context_dim=c4_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.down_4    = nn.Conv2d(channels*8,channels*16,**down_params)
		self.encoder_5 = ConditionedViTBlock(channels*16,num_heads=8,time_dim=time_dim,context_dim=c5_dim,mlp_ratio=mlp_ratio,depth=vit_layers)	

	def forward(self,x,t_emb,context):
		c3,c4,c5 = context
		enc_1 = self.encoder_1(x,t_emb)
		enc_2 = self.encoder_2(self.down_1(enc_1),t_emb)
		enc_3 = self.encoder_3(self.down_2(enc_2),t_emb,c3)
		enc_4 = self.encoder_4(self.down_3(enc_3),t_emb,c4)
		enc_5 = self.encoder_5(self.down_4(enc_4),t_emb,c5)
		return [enc_1,enc_2,enc_3,enc_4], enc_5


class ConditionEncoder(nn.Module):
	'''
	Encoder for RGB inputs.
	'''
	def __init__(self,in_channels=3,channels=32,cnn_layers=2):
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
		x  = self.stage_1(x)
		x  = self.stage_2(self.down_1(x))
		c3 = self.stage_3(self.down_2(x))
		c4 = self.stage_4(self.down_3(c3))
		c5 = self.stage_5(self.down_4(c4))
		return c3,c4,c5


################################################################################
# Decoder
################################################################################
class ViTDecoder(nn.Module):
	'''
	3-stage ViT & 2-stage CNN. Mirrors ViTEncoder().
	'''
	def __init__(self,cnn_layers=3,vit_layers=1,channels=32,mlp_ratio=4,time_dim=256,context_dims=(128,256,512)):
		super().__init__()	
		up_params = {'kernel_size': 4, 'stride': 2,'padding': 1, 'bias': True}
		c3_dim,c4_dim,c5_dim = context_dims

		self.decoder_1 = ConditionedViTBlock(channels*16,num_heads=8,time_dim=time_dim,context_dim=c5_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_1      = nn.ConvTranspose2d(channels*16,channels*8,**up_params)

		self.ch_mix_2  = nn.Conv2d(channels*16,channels*8,1,bias=True)
		self.decoder_2 = ConditionedViTBlock(channels*8,num_heads=4,time_dim=time_dim,context_dim=c4_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_2      = nn.ConvTranspose2d(channels*8,channels*4,**up_params)

		self.ch_mix_3  = nn.Conv2d(channels*8,channels*4,1,bias=True)
		self.decoder_3 = ConditionedViTBlock(channels*4,num_heads=2,time_dim=time_dim,context_dim=c3_dim,mlp_ratio=mlp_ratio,depth=vit_layers)
		self.up_3      = nn.ConvTranspose2d(channels*4,channels*2,**up_params)

		self.ch_mix_4  = nn.Conv2d(channels*4,channels*2,1,bias=True)
		self.decoder_4 = TimeConditionedConvBlock(channels*2,time_dim,depth=cnn_layers)
		self.up_4      = nn.ConvTranspose2d(channels*2,channels,**up_params)

		self.ch_mix_5  = nn.Conv2d(channels*2,channels,1,bias=True)
		self.decoder_5 = TimeConditionedConvBlock(channels,time_dim,depth=cnn_layers)


	def forward(self,x,skips,t_emb,context):
		enc_1,enc_2,enc_3,enc_4 = skips
		c3,c4,c5 = context
		dec_1 = self.decoder_1(x,t_emb,c5)
		dec_2 = self.decoder_2(self.ch_mix_2( torch.cat([enc_4,self.up_1(dec_1)],dim=1) ),t_emb,c4)
		dec_3 = self.decoder_3(self.ch_mix_3( torch.cat([enc_3,self.up_2(dec_2)],dim=1) ),t_emb,c3)
		dec_4 = self.decoder_4(self.ch_mix_4( torch.cat([enc_2,self.up_3(dec_3)],dim=1) ),t_emb)
		dec_5 = self.decoder_5(self.ch_mix_5( torch.cat([enc_1,self.up_4(dec_4)],dim=1) ),t_emb)
		return dec_5


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
# MODEL
################################################################################
class S2SegDiff(nn.Module):

	def __init__(self,model_id,mask_channels,in_channels,cnn_layers=3,vit_layers=1,channels=32,mlp_ratio=5,time_dim=256):
		super().__init__()

		self.model_name = "S2SegDiff"
		self.model_id   = model_id

		# time embeddings
		self.time_embedder = TimeEmbedding(channels,time_dim=time_dim)

		# context embeddings
		context_dims = (channels*4,channels*8,channels*16)
		self.cond_encoder  = ConditionEncoder(in_channels,channels,cnn_layers=2)

		# input projection layer to 'channels' argument
		self.in_layer = nn.Conv2d(mask_channels, channels, 3, 1, 1, bias=True)

		# encoder / decoder
		self.encoder = ViTEncoder(cnn_layers, vit_layers, channels, mlp_ratio,time_dim,context_dims)
		self.decoder = ViTDecoder(cnn_layers, vit_layers, channels, mlp_ratio,time_dim,context_dims)

		# final 1x1 classifier
		self.out_layer = nn.Conv2d(channels, mask_channels, kernel_size=1, bias=True)

	def forward(self,x_t,t,rgb_image):
		t_emb   = self.time_embedder(t)
		context = self.cond_encoder(rgb_image)

		x             = self.in_layer(x_t)
		skips,enc_out = self.encoder(x,t_emb,context)
		dec_out       = self.decoder(enc_out,skips,t_emb,context)
		return self.out_layer(dec_out)


################################################################################
# SOME UTILITY FUNCTIONS
################################################################################
def get_model_memory_footprint():
	pass

def get_model_parameter_size():
	pass

def count_flops():
	pass

################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	pass