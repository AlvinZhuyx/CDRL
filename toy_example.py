import os
import re
import torch
import time
import numpy as np 
import datetime as dt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import shutil
import random
from matplotlib import pyplot as plt
from torch_ema import ExponentialMovingAverage

################### hyper-parameter definition ###############################################################
seed = 1
log_path = './toy_example_res'
act_fn = 'lrelu'
N_data = 50000
N_sample = 25000
n_interval = 6
n_updates = 15
logsnr_min=-5.1
logsnr_max=9.8
latent_dim=100

n_iteration = 100000
n_save = 10000
n_print = 100
n_plot = 1000
p_ema_decay = 0.9999
pi_ema_decay=0.9999

t_dim = 128
e_h_dim = 128
p_h_dim = 64

e_n_layer = 1
p_n_layer = 1

pi_lr = 3e-6
p_lr = 1e-5
grad_clip = 1.0
variance_reduction = True
pred_var_type = 'small'
p_with_noise = True
grid_num = 200
sz_mul = 1.0
Langevin_clip = False
load_dir = None
load_path = None
test = False
mul = 4.0
x_min = -8.5
x_max = 8.5
use_sn = True
warmup_steps = 10000

sz_const = 1e-3
################### data generation ###############################################################################
def generate_square_samples(N):
    # return the samples from the squares in the list
    # list denote square representation (xmin, xmax, ymin, ymax)
    square_list = np.array([(1, 2, 0, 1), (0, 1, 1, 2), (0, 1, -1, 0), (1, 2, -2, -1), \
                            (-1, 0, -1, -2), (-2, -1, -1, 0), (-1, 0, 0, 1), (-2, -1, 1, 2)], dtype=np.float32) * mul

    idx = np.random.randint(low=0, high=len(square_list), size=(N,))
    s = square_list[idx]
    x_min, x_max, y_min, y_max = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    x_sample = (x_max - x_min) * np.random.uniform(low=0.0, high=1.0, size=(N,)) + x_min
    y_sample = (y_max - y_min) * np.random.uniform(low=0.0, high=1.0, size=(N,)) + y_min
    points = np.stack([x_sample, y_sample], axis=-1)
    return points

################### ebm model definition ##########################################################################
def get_timestep_embedding(timesteps, embedding_dim, dtype=torch.float32, max_time=1000.):
    assert len(timesteps.shape) == 1
    timesteps *= (1000. / max_time)
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(start=0, end=half_dim, dtype=dtype) * -np.log(10000) / (half_dim - 1)).cuda()
    emb = timesteps.type(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

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
            self.conv1 = spectral_norm(nn.Linear(in_ch, out_ch))
            self.conv2 = spectral_norm(nn.Linear(out_ch, out_ch))
            self.temb_map = spectral_norm(nn.Linear(temb_dim, out_ch))
        else:
            self.conv1 = nn.Linear(in_ch, out_ch)
            self.conv2 = nn.Linear(out_ch, out_ch)
            self.temb_map = nn.Linear(temb_dim, out_ch)
        
        self.g = nn.parameter.Parameter(torch.ones(1, out_ch), requires_grad=True)

        if in_ch != out_ch:
            if use_sn:
                self.short_cut = spectral_norm(nn.Linear(in_ch, out_ch)) 
            else:
                self.short_cut = nn.Linear(in_ch, out_ch)
        else:
            self.short_cut = nn.Identity()
        
    def forward(self, x, temb):        
        h = self.act_fn(x)
        h = self.conv1(h)
        temb_proj = self.act_fn(self.temb_map(self.act_fn(temb)))
        h = h + temb_proj
        h = self.act_fn(h)
        h = self.conv2(h) * self.g
        res = self.short_cut(x)
        return h + res


class ebm_mlp(nn.Module):
    def __init__(self, ch_mult=(1,2,2,2), in_channel=3, temb_dim=64, n_blocks=2, add_q=False, use_sn=False, act_fn='lrelu'):
        super(ebm_mlp, self).__init__()
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
        
        module_list = []
        if use_sn:
            self.temb_map1 = spectral_norm(nn.Linear(temb_dim, temb_dim*4))
            self.temb_map2 = spectral_norm(nn.Linear(temb_dim*4, temb_dim*4))
            module_list.append(spectral_norm(nn.Linear(in_channel, temb_dim)))
        else:
            self.temb_map1 = nn.Linear(temb_dim, temb_dim*4)
            self.temb_map2 = nn.Linear(temb_dim*4, temb_dim*4)
            module_list.append(nn.Linear(in_channel, temb_dim))
            
        in_ch = temb_dim
        for i_level in range(self.n_resolutions):
            for i_block in range(self.n_blocks):
                module_list.append(ResnetBlock(in_ch, temb_dim*ch_mult[i_level], temb_dim=4*temb_dim, use_sn=use_sn, ebm_act=act_fn))
                in_ch = temb_dim*ch_mult[i_level]
        
        self.module_list = nn.ModuleList(module_list)
        self.final_dense = nn.Linear(temb_dim*ch_mult[i_level], 1, bias=False)
    
    def forward(self, x, t):
        assert len(t) == len(x)
        logsnr_input = (torch.arctan(torch.exp(-0.5 * torch.clamp(t, min=-20., max=20.))) / (0.5 * np.pi))
        t_embed = get_timestep_embedding(logsnr_input, self.temb_dim, max_time=1.0)
        t_embed = self.act_fn(self.temb_map1(t_embed))
        t_embed = self.temb_map2(t_embed)

        h = self.module_list[0](x)
        layer_counter = 1
        for i_level in range(self.n_resolutions):
            for i_block in range(self.n_blocks):
                h = self.module_list[layer_counter](h, t_embed)
                layer_counter += 1
                
        h = self.final_act(h)
        #temb_final = self.temb_final(self.act_fn(t_embed))
        #print(h.shape, temb_final.shape)
        #h = torch.sum(h * temb_final, dim=1) #- self.final_bias(self.act_fn(t_embed)).squeeze()
        h = self.final_dense(h)
        h = torch.sum(h, dim=1)
        if self.add_q:
            add_q = -0.5 * torch.sum(x**2, dim=[1,2,3])
            h = h + add_q
        return h


################### unet model definition ##########################################################################        
class Unet_ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim = None):
        super().__init__()
        self.nonlinearity = F.silu #nn.SiLU()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6)
        self.updown = None
        self.conv1 = nn.Linear(dim, dim_out)
        self.conv2 = nn.Linear(dim_out, dim_out)
        self.conv2.weight.data.zero_()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.res_conv = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.nonlinearity(self.norm1(x))
        h = self.conv1(h)
    
        time_emb = self.mlp(time_emb)
        scale, shift = time_emb.chunk(2, dim = 1)
        
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.nonlinearity(h)
        h = self.conv2(h)

        return h + self.res_conv(x)

class unet_mlp(nn.Module):
    def __init__(self, t_dim, h_dim, n_layer):
        super(unet_mlp, self).__init__()
        self.temb_dim = t_dim
        self.act_fn = F.silu

        self.h1 = nn.Linear(2, h_dim)
        self.temb_map = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        ) 

        self.final_dense = nn.Linear(h_dim, 2, bias=False)
        self.final_dense.weight.data.zero_()
       
        module_list = []
        for i_level in range(n_layer):
            module_list.append(Unet_ResnetBlock(dim=h_dim, dim_out=h_dim, time_emb_dim=t_dim))
        self.module_list = nn.ModuleList(module_list)
    
    def forward(self, x, t):
        assert len(t) == len(x)
        logsnr_input = (torch.arctan(torch.exp(-0.5 * torch.clamp(t, min=-20., max=20.))) / (0.5 * np.pi))
        t_embed = get_timestep_embedding(logsnr_input, self.temb_dim, max_time=1.0)
        t_embed = self.temb_map(t_embed)
        h = self.h1(x)

        for layer in self.module_list:
            h = layer(h, t_embed)
                
        h = self.act_fn(h)
        h = self.final_dense(h)
        return h + x

################### diffusion function definition ##################################################################
def pred_x_from_eps(z, eps, logsnr):
    return torch.sqrt(1. + torch.exp(-logsnr)) * (z - eps * torch.rsqrt(1. + torch.exp(logsnr)))

def logsnr_schedule_fn(t, logsnr_min, logsnr_max):
    # -2log(tan(b)) == logsnr_max => b == arctan(exp(-0.5*logsnr_max))
    # -2log(tan(pi/2*a + b)) == logsnr_min
    #     => a == (arctan(exp(-0.5*logsnr_min))-b)*2/pi
    logsnr_min_tensor = logsnr_min * torch.ones_like(t)
    logsnr_max_tensor = logsnr_max * torch.ones_like(t)
    b = torch.arctan(torch.exp(-0.5 * logsnr_max_tensor))
    a = torch.arctan(torch.exp(-0.5 * logsnr_min_tensor)) - b
    return -2. * torch.log(torch.tan(a * t + b))

def diffusion_forward(x, logsnr):
    return {
        'mean': x * torch.sqrt(F.sigmoid(logsnr)),
        'std': torch.sqrt(F.sigmoid(-logsnr)),
        'var': F.sigmoid(-logsnr),
        'logvar': torch.log(F.sigmoid(-logsnr))
    }

def Langevin(pi, x, xt, logsnr_t, logsnr_tminus1, is_final, n_step=None, with_noise=True):
    if n_step is None:
        n_step = n_updates
    a_t = F.sigmoid(logsnr_t)
    a_tminus1 = F.sigmoid(logsnr_tminus1)
    as_t = (a_t / a_tminus1).reshape((len(x), 1))
    as_t_mul = as_t.detach().clone()
    as_t_mul[is_final > 0.0] = 1.0 
    sigma = torch.sqrt(1. - as_t)
    sigma_cum = (torch.sqrt(1. - a_tminus1))

    coeff = 1.0 / (1.0 + np.exp(-logsnr_max))

    ct_square = torch.ones_like(sigma_cum)  / np.sqrt(1.0 - coeff) 
    sz_square = (sz_const * ct_square * (sz_mul ** 2)).reshape((len(x), 1)) * sigma ** 2
    
    for i in range(n_step):
        en = pi(x, logsnr_tminus1)
        en = (en - torch.sum((x - xt) ** 2, dim=[1]) * ct_square * sz_const * (1.0 - is_final)) * (sz_mul ** 2)
        grad = torch.autograd.grad(en.sum(), [x])[0]
        x.data = x.data + 0.5 * grad 
        if with_noise:
            x.data = x.data + torch.sqrt(2. * sz_square) * torch.randn_like(x)

    x.data = x.data / (torch.sqrt(as_t_mul) + 1e-8)
    if Langevin_clip:
        x.data = torch.clamp(x.data, min=-1.0, max=1.0)
    x.requires_grad = False
    return x.detach()

def gen_samples(bs, p, pi, xt=None):
    if xt is None:
        xt = torch.randn(bs, 2).cuda()
    else:
        assert False
    mystring = 'step max/min: {} {:.2f}/{:.2f}'.format(n_interval, xt.max().item(), xt.min().item())
    xts = []
    for i in reversed(range(1, n_interval + 1)):
        i_tensor = torch.ones(bs, dtype=torch.float).cuda() * float(i)
        is_final = (i_tensor == n_interval).type(torch.float)
        logsnr_t = logsnr_schedule_fn(i_tensor / n_interval, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
        logsnr_s = logsnr_schedule_fn(torch.clamp(i_tensor - 1.0, min=0.0) / n_interval, logsnr_min=logsnr_min, logsnr_max=logsnr_max)

        a_t = F.sigmoid(logsnr_t)
        a_tminus1 = F.sigmoid(logsnr_s)

        sigma_cum = torch.sqrt(1. - a_tminus1)
        as_t = (a_t / a_tminus1)
        as_t_mul = as_t.detach().clone()
        as_t_mul[is_final > 0.0] = 1.0
        beta_t = 1 - as_t
        beta_t_tilt = (1.0 - a_tminus1) / (1.0 - a_t) * beta_t

        sigma = torch.sqrt(1. - as_t)
        coeff = 1.0 / (1.0 + np.exp(-logsnr_max))
        ct_square = torch.ones_like(sigma_cum) / np.sqrt(1.0 - coeff) 
        sz_square = (sz_const * ct_square * (sz_mul ** 2)) * sigma ** 2

        with torch.no_grad():
            if p_with_noise:
                xtminus1_neg0 = p(xt, logsnr_t) + torch.sqrt(beta_t_tilt * as_t_mul).reshape((len(xt), 1)) * torch.randn_like(xt)
            else:
                xtminus1_neg0 = p(xt, logsnr_t)

        xtminus1_negk = xtminus1_neg0.detach().clone()
        xtminus1_negk.requires_grad = True           
        xt = Langevin(pi, xtminus1_negk, xt, logsnr_t, logsnr_s, is_final, with_noise=True)
        mystring += ' {} {:.2f}/{:.2f}'.format(i - 1, xt.max().item(), xt.min().item())

        if i == -1:
            xt_in = xt.detach().clone()
            xt_in.requires_grad = True
            en = pi(xt_in * torch.sqrt(as_t_mul).reshape((len(xt), 1)), logsnr_s)
            en = en / sz_square
            score = torch.autograd.grad(en.sum(), xt_in)[0]
            xt = (xt_in.data + sigma_cum.reshape((len(xt), 1)) ** 2 * score) / torch.sqrt(a_tminus1).reshape((len(xt), 1))
            xt_in.requires_grad = False
        xts.append(xt.clone().detach().cpu().numpy())
    return xts

def visualize(x, pi):
    ens = []
    for i in reversed(range(1, n_interval + 1)):
        i_tensor = torch.ones(len(x), dtype=torch.float).cuda() * float(i)
        is_final = (i_tensor == n_interval).type(torch.float)
        logsnr_t = logsnr_schedule_fn(i_tensor / n_interval, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
        logsnr_tminus1 = logsnr_schedule_fn(torch.clamp(i_tensor - 1.0, min=0.0) / n_interval, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
        
        a_t = F.sigmoid(logsnr_t)
        a_tminus1 = F.sigmoid(logsnr_tminus1)
        as_t = (a_t / a_tminus1)
        as_t_mul = as_t.detach().clone()
        as_t_mul[is_final > 0.0] = 1.0
        sigma = torch.sqrt(1. - as_t)
        sigma_cum = (torch.sqrt(1. - a_tminus1))

        coeff = 1.0 / (1.0 + np.exp(-logsnr_max))
        ct_square = torch.ones_like(sigma_cum) / np.sqrt(1.0 - coeff)
        sz_square = (sz_const * ct_square * (sz_mul ** 2)).reshape((len(x))) * sigma ** 2
        
        energy =  pi(x *  torch.sqrt(as_t_mul.reshape((len(x), 1))), logsnr_tminus1) / (2 * sz_square)

        ens.append(energy.detach().cpu().numpy())
    return ens

def save_figures(samples, ens, save_path):
    assert len(ens) == n_interval
    assert len(samples) == n_interval
    fig, axes = plt.subplots(3, n_interval, figsize=(n_interval * 10, 30))
    # plot energy
    for i in range(n_interval):
        tmp_m = np.reshape(ens[i], (grid_num, grid_num))
        axes[0, i].matshow(tmp_m)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

    # plot probability
    for i in range(n_interval):
        tmp_m = np.reshape(ens[i], (grid_num, grid_num))
        tmp_m = tmp_m - tmp_m.max()
        tmp_m = np.exp(tmp_m)
        tmp_m = tmp_m / (np.sum(tmp_m) + 1e-6)
        axes[1, i].matshow(tmp_m)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    # plot samples
    for i in range(n_interval):
        s = samples[i]
        axes[2, i].scatter(s[:, 0], s[:, 1], s=0.5)
        axes[2, i].set_xlim(x_min, x_max)
        axes[2, i].set_ylim(x_min, x_max)
    
    fig.savefig(save_path + '_en_prob_s.png')
    plt.close(fig)



##################### training code ################################################################################
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# create the log path
timestamp = str(dt.datetime.now())[:19]
timestamp = re.sub(r'[\:-]','', timestamp) # replace unwanted chars 
timestamp = re.sub(r'[\s]','_', timestamp) # with regex and re.sub

if not test:
    save_path = os.path.join(log_path, timestamp)
    ckpt_path = os.path.join(save_path, 'ckpt')
    img_path = os.path.join(save_path, 'img')
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(log_path, timestamp, __file__))
data = generate_square_samples(N_data)

# define the model
pi = ebm_mlp(ch_mult=(1,2,2,2), in_channel=2, temb_dim=t_dim, n_blocks=e_n_layer, add_q=False, use_sn=use_sn, act_fn=act_fn)
p = unet_mlp(t_dim=t_dim, h_dim=p_h_dim, n_layer=p_n_layer)
pi.to('cuda')
p.to('cuda')

pi_ema = ExponentialMovingAverage(pi.parameters(), decay=pi_ema_decay)
p_ema = ExponentialMovingAverage(p.parameters(), decay=p_ema_decay)

pi_optimizer = optim.Adam(pi.parameters(), lr=pi_lr, betas=(0.9, 0.999))
p_optimizer = optim.AdamW(p.parameters(), lr=p_lr, betas=(0.9, 0.999), weight_decay=0.0)

p_lr_scheduler = optim.lr_scheduler.LambdaLR(p_optimizer, lr_lambda=lambda x: min(1.0, (x + 500) / float(warmup_steps)), last_epoch=-1, verbose=False)
pi_lr_scheduler = optim.lr_scheduler.LambdaLR(pi_optimizer, lr_lambda=lambda x: min(1.0, x / float(warmup_steps)), last_epoch=-1, verbose=False)

# get data for visualization
tmp_x = np.linspace(-2.0, 2.0, grid_num) * mul
tmp_y = np.linspace(2.0, -2.0, grid_num) * mul
plot_y, plot_x = np.meshgrid(tmp_y, tmp_x)
plot_x = np.reshape(plot_x, (grid_num ** 2, -1))
plot_y = np.reshape(plot_y, (grid_num ** 2, -1))
plot_point = np.concatenate([plot_y, plot_x], axis=-1)
plot_point = torch.tensor(plot_point, dtype=torch.float32).cuda()

data_tensor = torch.tensor(data, dtype=torch.float32)
fig, axes = plt.subplots(1, n_interval, figsize=(n_interval * 10, 10))
# visualize data at each distribution
for t in reversed(range(n_interval)):  
    t_tensor = torch.ones(len(data), dtype=torch.float) * float(t) / n_interval
    logsnr_t = logsnr_schedule_fn(t_tensor, logsnr_max=logsnr_max, logsnr_min=logsnr_min)
    zt_dist = diffusion_forward(data_tensor, logsnr=logsnr_t.reshape(len(data_tensor), 1))
    data_tensor_t =  zt_dist['mean'] + zt_dist['std'] * torch.randn_like(data_tensor)
    data_tensor_t = data_tensor_t.detach().numpy()
    axes[n_interval - 1 - t].scatter(data_tensor_t[:, 0], data_tensor_t[:, 1], s=0.5)
    axes[n_interval - 1 - t].set_xlim(x_min, x_max)
    axes[n_interval - 1 - t].set_ylim(x_min, x_max)
if not test:
    fig.savefig((os.path.join(img_path, 'obs.png')))
    plt.close(fig)

data_tensor = None

if test:
    print('loading from ', load_path)
    state_dict = torch.load(load_path)
    pi.load_state_dict(state_dict['pi_state_dict'])
    p.load_state_dict(state_dict['p_state_dict'])
    pi_ema.load_state_dict(state_dict['pi_ema_state_dict'])
    p_ema.load_state_dict(state_dict['p_ema_state_dict'])

    p.eval()
    pi.eval()

    samples = gen_samples(bs=N_sample, p=p, pi=pi)
    with torch.no_grad():
        ens = visualize(plot_point, pi)
    save_figures(samples, ens, save_path=os.path.join(load_dir, '{}_vis'.format('test')))
    with pi_ema.average_parameters():
        with p_ema.average_parameters():
            ema_samples = gen_samples(bs=N_sample, p=p, pi=pi)
            with torch.no_grad():
                ema_ens = visualize(plot_point, pi)
    save_figures(ema_samples, ema_ens, save_path=os.path.join(load_dir, '{}_ema'.format('test')))
    n_iteration = -1 # disable training


start_time = time.time()
for counter in range(n_iteration + 1):
    # sample training data
    idx = np.random.choice(len(data), size=(2500,), replace=False)
    x = torch.tensor(data[idx], dtype=torch.float).cuda()
    t = torch.randint(low=1, high=n_interval + 1, size=(len(x),), dtype=torch.float).cuda() / n_interval

    is_final = (t == 1.0).type(torch.float) 
    t_minus1 = t - 1.0 / n_interval
    assert torch.all(t_minus1 >= 0.0) 
    logsnr_t = logsnr_schedule_fn(t, logsnr_max=logsnr_max, logsnr_min=logsnr_min)
    logsnr_tminus1 = logsnr_schedule_fn(t_minus1, logsnr_max=logsnr_max, logsnr_min=logsnr_min)

    zt_dist = diffusion_forward(x, logsnr=logsnr_t.reshape(len(x), 1))
    ztminus1_dist = diffusion_forward(x, logsnr=logsnr_tminus1.reshape(len(x), 1))
    tmpe = torch.randn_like(x)
    xt = zt_dist['mean'] + zt_dist['std'] * tmpe
    xtminus1 = ztminus1_dist['mean'] + ztminus1_dist['std'] * tmpe

    # calculate true energy
    a_t = F.sigmoid(logsnr_t)
    a_tminus1 = F.sigmoid(logsnr_tminus1)
    as_t = (a_t / a_tminus1)
    as_t_mul = as_t.detach().clone()
    as_t_mul[is_final > 0.0] = 1.0
    
    a_1 = F.sigmoid(logsnr_schedule_fn(torch.tensor([1.0 / n_interval]), logsnr_max=logsnr_max, logsnr_min=logsnr_min))
    a_0 = F.sigmoid(logsnr_schedule_fn(torch.tensor([0.0]), logsnr_max=logsnr_max, logsnr_min=logsnr_min))
    sigma_1 = torch.sqrt(1.0 - a_1 / a_0).item()
    beta_t = 1 - as_t
    beta_t_tilt = (1.0 - a_tminus1) / (1.0 - a_t) * beta_t
    sigma_t = torch.sqrt(beta_t)

    sigma_cum = (torch.sqrt(1. - a_tminus1))
    coeff = 1.0 / (1.0 + np.exp(-logsnr_max))
    ct_square = torch.ones_like(sigma_cum) / np.sqrt(1.0 - coeff) #sigma_cum / np.sqrt(1.0 - coeff)
    sz_square = (sz_const * ct_square * (sz_mul ** 2)).reshape((len(x), 1)) * sigma_t ** 2

    # update pi
    pi_loss_weight = 1.0 / (sigma_t / sigma_1)
    pi.train()
    pos_energy = pi(xtminus1 *  torch.sqrt(as_t_mul.reshape((len(x), 1))), logsnr_tminus1) 
    pos_loss = -(pos_energy * pi_loss_weight).mean()
    pos_loss.backward()
    p.train()
    if p_with_noise:
        xtminus1_neg0 = p(xt, logsnr_t) + torch.sqrt(beta_t_tilt * as_t_mul).reshape((len(x), 1)) * torch.randn_like(xt)
    else:
        xtminus1_neg0 = p(xt, logsnr_t)
    xtminus1_negk = xtminus1_neg0.detach().clone()
    xtminus1_negk.requires_grad = True
    pi.eval()
    

    xtminus1_negk = Langevin(pi, xtminus1_negk, xt, logsnr_t, logsnr_tminus1, is_final)        
    neg_energy = pi(xtminus1_negk * torch.sqrt(as_t_mul.reshape((len(x), 1))), logsnr_tminus1)
    neg_loss = (neg_energy * pi_loss_weight).mean()
    neg_loss.backward()
    torch.nn.utils.clip_grad_norm(pi.parameters(), max_norm=grad_clip)
    pi_optimizer.step()
    pi_ema.update()
    pi_lr_scheduler.step()

    
    p_loss = torch.sum((xtminus1_negk * torch.sqrt(as_t_mul.reshape((len(x), 1))) - xtminus1_neg0) ** 2, dim=[1])
    p_loss = p_loss.mean()
    p_loss.backward()
    torch.nn.utils.clip_grad_norm(p.parameters(), max_norm=grad_clip)
    p_optimizer.step()
    p_ema.update()
    p_lr_scheduler.step()

    if counter % n_print == 0:
        print("Iter {} time{:.2f} pos en {:.2f} neg en {:.2f} pi loss {:.2f} p loss {:.2f}"\
            .format(counter, time.time() - start_time, -pos_loss.item(), neg_loss.item(), pos_loss.item() + neg_loss.item(), p_loss.item()))

    if counter % n_save == 0:
        print("save checkpoint")
        save_dict = {
            'pi_state_dict': pi.state_dict(),
            'p_state_dict': p.state_dict(),
            'pi_ema_state_dict': pi_ema.state_dict(),
            'p_ema_state_dict': p_ema.state_dict(),
            'pi_optimizer': pi_optimizer.state_dict(),
            'p_optimizer': p_optimizer.state_dict(),
        }
        torch.save(save_dict, os.path.join(ckpt_path, '{}.pth.tar'.format(counter)))
    
    if counter % n_plot == 0:
        p.eval()
        pi.eval()
        with pi_ema.average_parameters():
            with p_ema.average_parameters():
                ema_samples = gen_samples(bs=N_sample, p=p, pi=pi)
                with torch.no_grad():
                    ema_ens = visualize(plot_point, pi)
                save_figures(ema_samples, ema_ens, save_path=os.path.join(img_path, '{}_ema'.format(counter)))