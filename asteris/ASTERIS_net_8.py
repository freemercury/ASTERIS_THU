# ASTERIS: Pushing Detection Limits of Astronomical Imaging via 
# Self-supervised Spatiotemporal Denoising
# Author: Yuduo Guo, Hao Zhang, Mingyu Li
# Tsinghua University, Beijing, China

# ASTERIS is a deep learning framework for pushing the detection limit of astronomical 
# imaging, with a focus on spatiotemporal denoising across multi-exposure observations. 
# It is built upon and significantly extends the architecture of 
# [Restormer](https://arxiv.org/abs/2111.09881) by introducing temporal modeling and 
# adaptive restoration tailored for scientific image sequences.
# We sincerely thank the original authors of Restormer for making their code and design 
# publicly available.

# ## ⚖️ License & Copyright
# All original contributions, modifications, and extensions made in this project, 
# including the ASTERIS model and training framework, are copyrighted © 2025 by Yuduo Guo.
# This repository is released under the MIT License, unless otherwise specified. 
# See the [LICENSE](./LICENSE) file for details.
# ---
# ## ✉️ Contact
# For questions or potential collaborations, please contact Yuduo Guo at `gyd@mail.tsinghua.edu.cn`.
# Copyright (c) 2025 Yuduo Guo. All rights reserved.
# Date: 2025-05-22

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

def to_3d(x):
    # (b, c, d, h, w) -> (b, d*h*w, c)
    return rearrange(x, 'b c d h w -> b (d h w) c')

def to_4d(x,d,h,w):
    # (b, d*h*w, c) -> (b, c, d, h, w)
    return rearrange(x, 'b (d h w) c -> b c d h w',d=d,h=h,w=w)

##########################################################################
## Layer Norm
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # Accept int or tuple
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        # Scale only (no bias)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # Variance over last dim
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        # Accept int or tuple
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        # Scale and bias parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # Mean/variance over last dim
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        # Choose bias-free or with-bias variant
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # Flatten 3D dims to tokens -> apply LN -> restore shape
        d, h, w = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), d, h, w)

##########################################################################
## 3D Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        # Hidden width expansion
        hidden_features = int(dim*ffn_expansion_factor)
        # Pointwise expansion
        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)
        # Depthwise 3D conv
        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, padding_mode='replicate', groups=hidden_features*2, bias=bias)
        # Project back
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
###################################################################
##########################################################################
## 3D Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # Per-head temperature
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # Branch for general 3D
        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, padding_mode='replicate', groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        # Specialized branch for single-slice depth
        self.qkv2 = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out2 = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,d,h,w = x.shape

        if d != 1:      
            # 3D attention path      
            qkv = self.qkv_dwconv(self.qkv(x))
            q,k,v = qkv.chunk(3, dim=1)   
            # Split channels into heads and flatten spatial+depth to tokens
            q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
            # L2-normalize along token dim for stable dot products
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)# (b,h,c,c_tokens)
            # Aggregate values
            out = (attn @ v)    # (b,h,c,c) x (b,h,c,dhw) = (b,h,c,dhw)
            # Merge heads and restore 3D layout
            out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', 
                            head=self.num_heads, d=d, h=h, w=w)

            out = self.project_out(out)
        else:
            # 2D-on-3D degenerate path (single slice)
            qkv = self.qkv_dwconv2(self.qkv2(x))
            q,k,v = qkv.chunk(3, dim=1)   
            
            q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)

            out = (attn @ v)   
            
            out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', 
                            head=self.num_heads, d=d, h=h, w=w)

            out = self.project_out2(out)
            
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # Pre-norm attention + residual
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        # Pre-norm FFN + residual
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # Residual connections around attention and FFN
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        # Simple 3D conv to lift channels to embedding dim (stride=1 -> overlap)
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        # (b, in_c, d, h, w) -> (b, embed_dim, d, h, w)
        x = self.proj(x)
        return x

##########################################################################
## Resizing modules
class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor
    
    def forward(self, x):
        # Channel-to-space rearrangement in 3D
        batch_size, in_channels, depth, height, width = x.size()
        upscale_factor = self.upscale_factor
        # New channel count after distributing into 3 dims
        new_c = in_channels // (upscale_factor ** 3)
        # Split channels into 3D factors
        x = x.view(batch_size, new_c, upscale_factor, upscale_factor, upscale_factor, depth, height, width)
        # Interleave along D/H/W
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        # Combine to upscaled 3D volume
        x = x.view(batch_size, new_c, depth * upscale_factor, height * upscale_factor, width * upscale_factor)
        
        return x

class PixelUnshuffle3D(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle3D, self).__init__()
        self.downscale_factor = downscale_factor
    
    def forward(self, x):
        # Inverse PixelShuffle3D
        batch_size, in_channels, depth, height, width = x.size()
        downscale_factor = self.downscale_factor
        
        new_d = depth // downscale_factor
        new_h = height // downscale_factor
        new_w = width // downscale_factor
        new_c = in_channels * (downscale_factor ** 3)
        # Split D/H/W into (new_dim, factor)
        x = x.view(batch_size, in_channels, new_d, downscale_factor, new_h, downscale_factor, new_w, downscale_factor)
        
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        # Merge factors into channels
        x = x.view(batch_size, new_c, new_d, new_h, new_w)
        
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # Reduce channels, then unshuffle
        pixel_unshuffle_3d = PixelUnshuffle3D(downscale_factor=2)
        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=1, bias=False),pixel_unshuffle_3d)

    def forward(self, x):
        x = self.body(x)
        return x
        
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        # Shuffle to double D/H/W after expanding channels
        pixel_shuffle_3d = PixelShuffle3D(upscale_factor=2)      
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
            pixel_shuffle_3d)

    def forward(self, x):        
        x = self.body(x)
        return x

class TransformerAttention3D(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, nhead=4):
        super(TransformerAttention3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead
        # Linear projection to Q/K/V along channel dim
        self.qkv_projection = nn.Linear(out_channels, max(out_channels, nhead) * 3)
        # Multi-head attention over flattened H*W per depth step
        self.attention = nn.MultiheadAttention(embed_dim=max(out_channels, nhead), num_heads=nhead, bias=bias)
        # 2D conv to mix H/W per depth slice
        self.hw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # Project attention output back to out_channels
        self.output_linear = nn.Linear(max(out_channels, nhead), out_channels)

    def forward(self, x):
        batch_size, channels, d, h, w = x.shape
        # Apply 2D conv per depth slice to produce features
        hw_output = self.hw_conv(x.view(batch_size * d, channels, h, w))  # (B*d, out_c, H, W)
        hw_output = hw_output.view(batch_size, d, self.out_channels, h, w).permute(0, 2, 1, 3, 4)  # (B, out_c, d, H, W)
        # Flatten (H,W) into tokens per depth, attention over tokens with channel as embedding
        x_attn = hw_output.permute(2, 0, 3, 4, 1).reshape(d, -1, self.out_channels)
        # Build Q/K/V and run MHA
        qkv = self.qkv_projection(x_attn)  # Shape: (d, batch_size * h * w, max(out_channels, nhead) * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Split last dimension into 3 parts
        attn_output, _ = self.attention(q, k, v)
        
        # Map back to out_channels and restore (B, C, d, H, W)
        attn_output = self.output_linear(attn_output)  # Shape: (d, batch_size * h * w, out_channels)
        attn_output = attn_output.reshape(d, batch_size, h, w, self.out_channels).permute(1, 4, 0, 2, 3)

        return attn_output

##########################################################################
##---------- ASTERIS -----------------------
class ASTERIS8(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        f_maps = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'BiasFree',   ## Other option 'BiasFree' 'WithBias'
    ):

        super(ASTERIS8, self).__init__()
        
        dim = f_maps
        # Initial embedding 
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        # Encoder stage 1
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # Downsample to stage 2
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        # Downsample to stage 3
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        # Downsample to stage 4
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        # Decoder from stage 4 -> 3
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        # Decoder from stage 3 -> 2
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        # Decoder from stage 2 -> 1
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # Extra refinement at highest resolution
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        # Final 3D attention head + residual connection to input  
        self.output = TransformerAttention3D(int(dim*2**1), out_channels)

    def forward(self, inp_img):
        # ---------------- Encoder ----------------
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
        # ---------------- Decoder ----------------             
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img     

        return out_dec_level1