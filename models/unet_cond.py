# -*- coding: utf-8 -*-
# adapted unet implementation from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
import copy
from random import random
from functools import partial
from collections import namedtuple
from tkinter import W
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def l2norm(t):
    return F.normalize(t, dim = -1)

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample():
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest')
    )

def Downsample():
    return nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_time=1000.):
        super().__init__()
        self.dim = dim
        self.max_time = max_time

    def forward(self, x):
        x *= (1000. / self.max_time)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, resample=None, dropout=0.1):
        super().__init__()
        self.nonlinearity = F.silu 
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6)
        self.updown = None
        if resample is not None:
            if resample == 'up':
                self.updown = Upsample()
            elif resample == 'down':
                self.updown = Downsample()
            else:
                raise NotImplementedError
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2.weight.data.zero_()
        self.dropout_layer = nn.Dropout(p=dropout)

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.nonlinearity(self.norm1(x))
        if self.updown is not None:
            h = self.updown(h)
            x = self.updown(x)
        h = self.conv1(h)
    
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1').contiguous()
            scale, shift = time_emb.chunk(2, dim = 1)
        
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.nonlinearity(h)
        h = self.dropout_layer(h)
        h = self.conv2(h)

        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = None, dim_head = None):
        super().__init__()
        print('Using attention')
        if dim_head is None:
            assert heads is not None
            assert dim % heads == 0
            dim_head = dim // heads

        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = True)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.to_out.weight.data.zero_()

    def forward(self, x):
        b, c, height, width = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads).contiguous(), qkv)
        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = height, y = width).contiguous()
        return self.to_out(out) + x

class Unet(nn.Module):
    def __init__(
        self,
        dim = 256,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 1, 1),
        channels = 3,
        im_sz = 32,
        attn_resolutions=[8, 16],
        num_res_blocks=3, 
        dropout=0.1,
        num_heads=1,
        tanh_out = True,
        residual = True,
        n_class = None,
        p_cond = None
    ):
        super().__init__()
        self.tanh_out = tanh_out
        self.residual = residual
        # determine dimensions
        self.im_sz = im_sz
        self.channels = channels
        self.attn_resolutions = attn_resolutions
        self.num_res_blocks = num_res_blocks
        self.n_class_emb = n_class + 1 # class 0 corresponds to emtpy class
        self.p_cond = p_cond

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, kernel_size=3, stride=1, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim, max_time=1.)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.y_mlp = nn.Sequential(
            nn.Linear(self.n_class_emb, time_dim)
        )
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        cur_size = im_sz

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            current_down = nn.ModuleList([])
            current_down.append(ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, resample=None, dropout=dropout))
            if cur_size in attn_resolutions:
                current_down.append(Attention(dim_out, heads=num_heads)) 
            for i_block in range(num_res_blocks - 1):
                current_down.append(
                    ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, resample=None, dropout=dropout)
                ) 
                if cur_size in attn_resolutions:
                    current_down.append(Attention(dim_out, heads=num_heads))
            if not is_last:
                current_down.append(ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, resample='down', dropout=dropout))
                cur_size = cur_size // 2
            self.downs.append(current_down)


        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, resample=None, dropout=dropout)
        self.mid_attn = Attention(mid_dim, heads=num_heads)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, resample=None, dropout=dropout)


        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            current_up = nn.ModuleList([])

            current_up.append(ResnetBlock(dim_in * 2, dim_out, time_emb_dim=time_dim, resample=None, dropout=dropout))
            if cur_size in attn_resolutions:
                current_up.append(Attention(dim_out, heads=num_heads)) 
            for i_block in range(num_res_blocks):
                current_up.append(
                    ResnetBlock(dim_out * 2, dim_out, time_emb_dim=time_dim, resample=None, dropout=dropout)
                ) 
                if cur_size in attn_resolutions:
                    current_up.append(Attention(dim_out, heads=num_heads))
            if not is_last:
                current_up.append(ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, resample='up', dropout=dropout))
                cur_size = cur_size * 2
            self.ups.append(current_up)

        assert len(self.downs) == num_resolutions
        assert len(self.ups) == num_resolutions
        
        self.nonlinearity = F.silu #nn.SiLU()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6)

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, kernel_size=3, stride=1, padding=1)
        self.final_conv.weight.data.zero_()

    def forward(self, x, logsnr, y):
        b, c, h, w = x.shape
        assert h == w
        assert h == self.im_sz
        assert logsnr.shape == (b,) 
        assert y.shape == (b,)      

        logsnr_input = (torch.arctan(torch.exp(-0.5 * torch.clamp(logsnr, min=-20., max=20.))) / (0.5 * np.pi))
        t = self.time_mlp(logsnr_input)
        y_one_hot = F.one_hot(y, self.n_class_emb)
        y_emb = self.y_mlp(y_one_hot.float())
        t = t + y_emb

        h = self.init_conv(x)
        hs = [h]
        
        num_resolutions = len(self.downs)
        
        for ind in range(num_resolutions):
            current_down = self.downs[ind]
            is_last = ind >= (num_resolutions - 1)
            counter = 0
            for i_block in range(self.num_res_blocks):
                h = current_down[counter](x=hs[-1], time_emb=t) # resblock
                counter += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = current_down[counter](h) # attention
                    counter += 1
                hs.append(h)

            if not is_last:
                hs.append(current_down[counter](hs[-1], time_emb=t))
                counter += 1

            assert counter == len(current_down)
        
        h = hs[-1]
        h = self.mid_block1(h, time_emb=t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb=t)

        for ind in range(num_resolutions):
            is_last = ind >= (num_resolutions - 1) 
            current_up = self.ups[ind]
            counter = 0
            for i_block in range(self.num_res_blocks + 1):
                h = current_up[counter](torch.cat([h, hs.pop()], dim=1), time_emb=t) # resblock
                counter += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = current_up[counter](h) # attention
                    counter += 1

            if not is_last:
                h = current_up[counter](h, time_emb=t)
                counter += 1
            assert counter  == len(current_up)
        
        assert len(hs) == 0
        h = self.nonlinearity(self.norm(h))
        out = self.final_conv(h)
        if self.residual:
            out = x + out

        if self.tanh_out:
            return torch.tanh(out)
        else:
            return out