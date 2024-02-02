# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

#################################### EBM with time embedding and class condition ########################################

class EBM(nn.Module):
    def __init__(self, ch_mult=(1,2,2,2), in_channel=3, temb_dim=64, n_blocks=2, add_q=False, use_sn=False, act_fn='lrelu', n_class=None, p_cond=None):
        super(EBM, self).__init__()
        self.add_q = add_q
        if act_fn == 'lrelu':
            self.act_fn = lambda x: F.leaky_relu(x, negative_slope=0.2)
            self.final_act = F.relu
        elif act_fn == 'swish':
            self.act_fn = lambda x: F.silu(x)
            self.final_act = lambda x: F.silu(x)
        else:
            raise NotImplementedError

        self.n_resolutions = len(ch_mult)
        self.n_blocks = n_blocks
        self.temb_dim = temb_dim

        self.n_class_emb = n_class + 1 # class 0 corresponds to emtpy class
        self.p_cond = p_cond
        
        module_list = []
        if use_sn:
            self.temb_map1 = spectral_norm(nn.Linear(temb_dim, temb_dim*4))
            self.temb_map2 = spectral_norm(nn.Linear(temb_dim*4, temb_dim*4))
            self.y_map = spectral_norm(nn.Linear(self.n_class_emb, temb_dim*4))
            module_list.append(spectral_norm(nn.Conv2d(in_channel, temb_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)))
        else:
            self.temb_map1 = nn.Linear(temb_dim, temb_dim*4)
            self.temb_map2 = nn.Linear(temb_dim*4, temb_dim*4)
            self.y_map = nn.Linear(self.n_class_emb, temb_dim*4)
            module_list.append(nn.Conv2d(in_channel, temb_dim, kernel_size=(3, 3), stride=(1, 1), padding=1))
        
            
        in_ch = temb_dim
        for i_level in range(self.n_resolutions):
            for i_block in range(self.n_blocks):
                module_list.append(ResnetBlock(in_ch, temb_dim*ch_mult[i_level], temb_dim=4*temb_dim, use_sn=use_sn, ebm_act=act_fn))
                in_ch = temb_dim*ch_mult[i_level]
            if i_level != self.n_resolutions - 1:
                module_list.append(nn.AvgPool2d((2, 2), (2, 2)))
        
        self.module_list = nn.ModuleList(module_list)
                
        self.temb_final = nn.Linear(4*temb_dim, temb_dim*ch_mult[i_level], bias=False)

    def forward(self, x, t, y):
        assert len(t) == len(x)
        logsnr_input = (torch.arctan(torch.exp(-0.5 * torch.clamp(t, min=-20., max=20.))) / (0.5 * np.pi))
        t_embed = get_timestep_embedding(logsnr_input, self.temb_dim, max_time=1.0)
        t_embed = self.act_fn(self.temb_map1(t_embed))
        t_embed = self.temb_map2(t_embed)
        y_one_hot = F.one_hot(y, self.n_class_emb)
        y_emb = self.y_map(y_one_hot.float())
        t_embed = t_embed + y_emb

        h = self.module_list[0](x)
        layer_counter = 1
        for i_level in range(self.n_resolutions):
            for i_block in range(self.n_blocks):
                h = self.module_list[layer_counter](h, t_embed)
                layer_counter += 1
                
            if i_level != self.n_resolutions - 1:
                h = self.module_list[layer_counter](h)
                layer_counter += 1
                
        h = self.final_act(h)
        h = torch.sum(h, dim=(2,3))
        temb_final = self.temb_final(self.act_fn(t_embed))
        h = torch.sum(h * temb_final, dim=1) 
        if self.add_q:
            add_q = -0.5 * torch.sum(x**2, dim=[1,2,3])
            h = h + add_q

        return h
        
class ResnetBlock(nn.Module):
    """Convolutional residual block with two convs."""
    def __init__(self, in_ch, out_ch, temb_dim, use_sn=False, ebm_act=None):
        super(ResnetBlock, self).__init__()
        self.out_ch = out_ch
        
        if ebm_act == 'lrelu':
            self.act_fn = lambda x: F.leaky_relu(x, negative_slope=0.2)
        elif ebm_act == 'swish':
             self.act_fn = lambda x: F.silu(x)
        else:
            raise NotImplementedError
        if use_sn:
            self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1))
            self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1))
            self.temb_map = spectral_norm(nn.Linear(temb_dim, out_ch))
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.temb_map = nn.Linear(temb_dim, out_ch)
        
        self.g = nn.parameter.Parameter(torch.ones(1, out_ch, 1, 1), requires_grad=True)
        if in_ch != out_ch:
            if use_sn:
                self.short_cut = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)) 
            else:
                self.short_cut = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)
        else:
            self.short_cut = nn.Identity()
        
    def forward(self, x, temb):        
        h = self.act_fn(x)
        h = self.conv1(h)
        temb_proj = self.act_fn(self.temb_map(self.act_fn(temb)))
        h = h + temb_proj[:, :, None, None]
        h = self.act_fn(h)
        h = self.conv2(h) * self.g
        res = self.short_cut(x)
        return h + res

def get_timestep_embedding(timesteps, embedding_dim, dtype=torch.float32, max_time=1000.):

    assert len(timesteps.shape) == 1
    timesteps *= (1000. / max_time)
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(start=0, end=half_dim, dtype=dtype) * -np.log(10000) / (half_dim - 1)).to(timesteps.device)
    emb = timesteps.type(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
