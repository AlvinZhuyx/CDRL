# -*- coding: utf-8 -*-

from statistics import variance
import argparse
import numpy as np
import os
import torch
import random
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import datetime as dt
import re
import shutil
import math
from models.ebm_cond import EBM
from models.unet_cond import Unet
from matplotlib import pyplot as plt
from torch_ema import ExponentialMovingAverage
import pytorch_fid_wrapper as pfw
from torch.nn.parallel import DistributedDataParallel as DDP
torch.multiprocessing.set_sharing_strategy('file_system')

########################## hyper parameters ###################################
seed = 5
dataset_type = 'Imagenet' 

if dataset_type == 'Imagenet':
    batch_size = 32
    c = 3
    im_sz = 32
    n_class = 1000
else:
    raise NotImplementedError

p_cond = 0.1
eval_weight = 1.0

n_interval = 6
n_updates = 15
logsnr_min=-5.1
logsnr_max=9.8
latent_dim=100

pred_var_type = 'small' # 'large' or 'small'
Langevin_clip = False
with_pixel_norm = True
p_with_noise = True
log_path = './logs/{}'.format(dataset_type)


mode = 'train' # 'train', 'fid_vs_w'

ckpt_idx = 400000
load_path = './logs/Imagenet/cond_best.pth.tar'

stats_path = None
p_lr = 1e-5
pi_lr = 1e-4
warmup_steps = 10000
p_weight_decay = 0.0
iterations = 1000001
p_ema_decay = 0.9999
pi_ema_decay=0.9999
grad_clip = 1.0
add_q = False
with_sn = True # whether use spectral norm in model
sn_step = 1
ebm_act = 'lrelu'

max_pretrain_pi_iter = 100001
print_iter = 100
plot_iter = 1000
ckpt_iter = 50000
fid_iter = 20000
n_fid_samples = 5000
n_fid_samples_full = 5000

sz_mul = 1.0
eta = 1.0 # add a coefficient to rescale the ratio term --> enable gradient to flow through

pi_n_blocks = 8 # number of blocks in pi
pi_dim = 128
p_dim = 64 # dimension for p
variance_reduction = True

#################### define a customized utils function #######################
class log1mexp(torch.autograd.Function):
    # From James Townsend's PixelCNN++ code
    # Method from
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.where(input > np.log(2.), torch.log1p(-torch.exp(-input)), torch.log(-torch.expm1(-input)))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output / torch.expm1(input)

mylog1mexp = log1mexp.apply

class tracker():
    def __init__(self, interval):
        self.record = []
        self.interval = interval

    def update(self, val):
        self.record.append(val)
    
    def plot(self, name):
        plt.figure()
        plt.plot(np.arange(len(self.record)) * self.interval, np.array(self.record))
        plt.savefig(name)
        plt.close()

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
    #print(a[0], b[0], torch.exp(-0.5 * logsnr_max_tensor[0]), torch.exp(-0.5 * logsnr_min_tensor[0]))
    return -2. * torch.log(torch.tan(a * t + b))

def diffusion_reverse(x, z_t, logsnr_s, logsnr_t):
    alpha_st = torch.sqrt((1. + torch.exp(-logsnr_t)) / (1. + torch.exp(-logsnr_s)))
    alpha_s = torch.sqrt(F.sigmoid(logsnr_s))
    r = torch.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
    one_minus_r = -torch.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
    log_one_minus_r = mylog1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))
    mean = r * alpha_st * z_t + one_minus_r * alpha_s * x
    if pred_var_type == 'large':
        var = one_minus_r * F.sigmoid(-logsnr_t)
        logvar = log_one_minus_r + torch.log(F.sigmoid(-logsnr_t))
    elif pred_var_type == 'small':
        a_t = F.sigmoid(logsnr_t)
        a_tminus1 = F.sigmoid(logsnr_s)
        beta_t = (1 - a_t / a_tminus1)
        var = (1.0 - a_tminus1) / (1.0 - a_t) * beta_t
        logvar = torch.log(var)
    else:
        raise NotImplemented
    return {'mean': mean, 'std': torch.sqrt(var), 'var': var, 'logvar': logvar}

def diffusion_forward(x, logsnr):
    return {
        'mean': x * torch.sqrt(F.sigmoid(logsnr)),
        'std': torch.sqrt(F.sigmoid(-logsnr)),
        'var': F.sigmoid(-logsnr),
        'logvar': torch.log(F.sigmoid(-logsnr))
    }

def gen_samples(bs, p, pi, xt=None, device=None):
    assert device is not None
    if xt is None:
        xt = torch.randn(bs, c, im_sz, im_sz).to(device)
        y = torch.randint(low=0, high=n_class + 1, size=(bs,)).to(device)
    else:
        assert False
    mystring = 'step max/min: {} {:.2f}/{:.2f}'.format(n_interval, xt.max().item(), xt.min().item())
    for i in reversed(range(1, n_interval + 1)):
        i_tensor = torch.ones(bs, dtype=torch.float).to(device) * float(i)
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
        ct_square = sigma_cum / np.sqrt(1.0 - coeff)
        sz_square = (2e-4 * ct_square * (sz_mul ** 2)) * sigma ** 2

        with torch.no_grad():
            if p_with_noise:
                xtminus1_neg0 = p(xt, logsnr_t, y) + torch.sqrt(beta_t_tilt * as_t_mul).reshape((len(xt), 1, 1, 1)) * torch.randn_like(xt)
            else:
                xtminus1_neg0 = p(xt, logsnr_t, y)

        xtminus1_negk = xtminus1_neg0.detach().clone()
        xtminus1_negk.requires_grad = True            
        xt = Langevin(pi, xtminus1_negk, xt, logsnr_t, logsnr_s, is_final, with_noise=True, y=y)
        mystring += ' {} {:.2f}/{:.2f}'.format(i - 1, xt.max().item(), xt.min().item())

        if i == 1:
            xt_in = xt.detach().clone()
            xt_in.requires_grad = True
            en = pi(xt_in * torch.sqrt(as_t_mul).reshape((len(xt), 1, 1, 1)), logsnr_s, y=y)
            en = en / sz_square
            score = torch.autograd.grad(en.sum(), xt_in)[0]
            xt = (xt_in.data + sigma_cum.reshape((len(xt), 1, 1, 1)) ** 2 * score) / torch.sqrt(a_tminus1).reshape((len(xt), 1, 1, 1))
            xt_in.requires_grad = False
    if device == torch.device('cuda', 0):
        print(mystring)
    return xt            
        
def gen_guided_samples(bs, p, pi, xt=None, guided_weight=None, device=None):
    assert device is not None
    _n_class = min(n_class, 10)
    assert bs % _n_class == 0
    if xt is None:
        xt = torch.randn(bs, c, im_sz, im_sz).to(device)
        n_sample_per_class = bs // _n_class
        y = torch.arange(1, _n_class+1).unsqueeze(1).repeat((1, n_sample_per_class)).reshape((bs,)).to(device)
        y_uncond = torch.zeros((bs,), dtype=torch.long).to(device)
    else:
        assert False
    mystring = 'step max/min: {} {:.2f}/{:.2f}'.format(n_interval, xt.max().item(), xt.min().item())
    #xts = []
    for i in reversed(range(1, n_interval + 1)):
        i_tensor = torch.ones(bs, dtype=torch.float).to(device) * float(i)
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
        ct_square = sigma_cum / np.sqrt(1.0 - coeff)
        sz_square = (2e-4 * ct_square * (sz_mul ** 2)) * sigma ** 2

        with torch.no_grad():
            pred =  (1.0 + guided_weight) * p(xt, logsnr_t, y=y) - guided_weight * p(xt, logsnr_t, y=y_uncond)
            if p_with_noise:
                xtminus1_neg0 = pred + torch.sqrt(beta_t_tilt * as_t_mul).reshape((len(xt), 1, 1, 1)) * torch.randn_like(xt)
            else:
                xtminus1_neg0 = pred
            

        xtminus1_negk = xtminus1_neg0.detach().clone()
        xtminus1_negk.requires_grad = True            
        xt = Langevin_guided(pi, xtminus1_negk, xt, logsnr_t, logsnr_s, is_final, with_noise=True, y=y, y_uncond=y_uncond, guided_weight=guided_weight)
        mystring += ' {} {:.2f}/{:.2f}'.format(i - 1, xt.max().item(), xt.min().item())

        if i == 1:
            xt_in = xt.detach().clone()
            xt_in.requires_grad = True
            en = (1.0 + guided_weight) * pi(xt_in * torch.sqrt(as_t_mul).reshape((len(xt), 1, 1, 1)), logsnr_s, y=y) \
                - guided_weight * pi(xt_in * torch.sqrt(as_t_mul).reshape((len(xt), 1, 1, 1)), logsnr_s, y=y_uncond)
            en = en / sz_square
            score = torch.autograd.grad(en.sum(), xt_in)[0]
            xt = (xt_in.data + sigma_cum.reshape((len(xt), 1, 1, 1)) ** 2 * score) / torch.sqrt(a_tminus1).reshape((len(xt), 1, 1, 1))
            xt_in.requires_grad = False
    if device == torch.device('cuda', 0):
        print(mystring)
    return xt

def gen_guided_samples2(bs, p, pi, xt=None, guided_weight=None, clip_langevin=False, device=None):
    assert device is not None
    if xt is None:
        xt = torch.randn(bs, c, im_sz, im_sz).to(device)
        y = torch.randint(low=1, high=n_class+1, size=(bs,)).to(device)
        y_uncond = torch.zeros((bs,), dtype=torch.long).to(device)
    else:
        assert False
    mystring = 'step max/min: {} {:.2f}/{:.2f}'.format(n_interval, xt.max().item(), xt.min().item())
    for i in reversed(range(1, n_interval + 1)):
        i_tensor = torch.ones(bs, dtype=torch.float).to(device) * float(i)
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
        ct_square = sigma_cum / np.sqrt(1.0 - coeff)
        sz_square = (2e-4 * ct_square * (sz_mul ** 2)) * sigma ** 2

        with torch.no_grad():
            pred =  (1.0 + guided_weight) * p(xt, logsnr_t, y=y) - guided_weight * p(xt, logsnr_t, y=y_uncond)
            if p_with_noise:
                xtminus1_neg0 = pred + torch.sqrt(beta_t_tilt * as_t_mul).reshape((len(xt), 1, 1, 1)) * torch.randn_like(xt)
            else:
                xtminus1_neg0 = pred

        xtminus1_negk = xtminus1_neg0.detach().clone()
        xtminus1_negk.requires_grad = True            
        xt = Langevin_guided(pi, xtminus1_negk, xt, logsnr_t, logsnr_s, is_final, with_noise=True, y=y, y_uncond=y_uncond, guided_weight=guided_weight, clip=clip_langevin)
        mystring += ' {} {:.2f}/{:.2f}'.format(i - 1, xt.max().item(), xt.min().item())

        if i == 1:
            xt_in = xt.detach().clone()
            xt_in.requires_grad = True
            en = (1.0 + guided_weight) * pi(xt_in * torch.sqrt(as_t_mul).reshape((len(xt), 1, 1, 1)), logsnr_s, y=y) \
                - guided_weight * pi(xt_in * torch.sqrt(as_t_mul).reshape((len(xt), 1, 1, 1)), logsnr_s, y=y_uncond)
            en = en / sz_square
            score = torch.autograd.grad(en.sum(), xt_in)[0]
            xt = (xt_in.data + sigma_cum.reshape((len(xt), 1, 1, 1)) ** 2 * score) / torch.sqrt(a_tminus1).reshape((len(xt), 1, 1, 1))
            xt_in.requires_grad = False

    if device == torch.device('cuda', 0):
        print(mystring)
    return xt

def calculate_fid(n_samples, p, pi, real_m, real_s, save_name=None, return_samples=False, device=None):
    assert device is not None
    if device == torch.device('cuda', 0):
        print('calculate fid')
    start_time = time.time()
    bs = 100
    fid_samples = []
    
    for i in range(n_samples // bs):
        cur_samples = gen_samples(bs, p, pi, device=device)
        fid_samples.append(cur_samples.detach().clone())
        if device == torch.device('cuda', 0):
            print("Generate {} samples, time {:.2f}".format((i+1) * bs, time.time() - start_time))

    fid_samples = torch.cat(fid_samples, dim=0)
    fid_samples = (1.0 + torch.clamp(fid_samples, min=-1.0, max=1.0)) / 2.0
    fid = pfw.fid(fid_samples, real_m=real_m, real_s=real_s, device=device)
    if save_name is not None:
        save_images = fid_samples[:100].clone().detach().cpu()
        torchvision.utils.save_image(save_images, save_name, normalize=True, nrow=10)
    if not return_samples:
        return fid
    else:
        return fid, fid_samples

def denoise_true(z, x0, logsnr_t, logsnr_tminus1):
    z_tminus1_dist = diffusion_reverse(x=x0, z_t=z, logsnr_s=logsnr_tminus1.reshape(len(z), 1, 1, 1), logsnr_t=logsnr_t.reshape(len(z), 1, 1, 1))
    a_t = F.sigmoid(logsnr_t)
    a_tminus1 = F.sigmoid(logsnr_tminus1)
    beta_t = (1 - a_t / a_tminus1)
    std = torch.sqrt((1.0 - a_tminus1) / (1.0 - a_t) * beta_t).reshape((len(z), 1, 1, 1))
    sample_x = z_tminus1_dist['mean'] + std * torch.randn_like(z)


    if torch.any(torch.isinf(sample_x)):
        print('logsnr_tminus1', logsnr_tminus1)
        print('logsnr_t', logsnr_t)
        print('a_t', a_t)
        print('a_tminus1', a_tminus1)
        print('beta_t', beta_t)
        print('std', std)
        if torch.any(torch.isinf(z_tminus1_dist['mean'])):
            print('inf in z_tmins1_dist')

        assert False
    return sample_x

def Langevin(pi, x, xt, logsnr_t, logsnr_tminus1, is_final, n_step=None, with_noise=True, y=None):
    if n_step is None:
        n_step = n_updates
    a_t = F.sigmoid(logsnr_t)
    a_tminus1 = F.sigmoid(logsnr_tminus1)
    as_t = (a_t / a_tminus1).reshape((len(x), 1, 1, 1))
    as_t_mul = as_t.detach().clone()
    as_t_mul[is_final > 0.0] = 1.0 
    sigma = torch.sqrt(1. - as_t)
    sigma_cum = (torch.sqrt(1. - a_tminus1))

    coeff = 1.0 / (1.0 + np.exp(-logsnr_max))

    ct_square = sigma_cum / np.sqrt(1.0 - coeff)
    sz_square = (4e-4 * ct_square * (sz_mul ** 2)).reshape((len(x), 1, 1, 1)) * sigma ** 2
    
    for i in range(n_step):
        en = pi(x, logsnr_tminus1, y=y)
        en = (en - torch.sum((x - xt) ** 2, dim=[1,2,3]) * ct_square * 2e-4 * (1.0 - is_final)) * (sz_mul ** 2)
        grad = torch.autograd.grad(en.sum(), [x])[0]
        x.data = x.data + 0.5 * grad 
        if with_noise:
            x.data = x.data + torch.sqrt(sz_square) * torch.randn_like(x)

    x.data = x.data / (torch.sqrt(as_t_mul) + 1e-8)
    if Langevin_clip:
        x.data = torch.clamp(x.data, min=-1.0, max=1.0)
    x.requires_grad = False
    return x.detach()

def Langevin_guided(pi, x, xt, logsnr_t, logsnr_tminus1, is_final, n_step=None, with_noise=True, y=None, y_uncond=None, guided_weight=None, clip=False):
    if n_step is None:
        n_step = n_updates
    a_t = F.sigmoid(logsnr_t)
    a_tminus1 = F.sigmoid(logsnr_tminus1)
    as_t = (a_t / a_tminus1).reshape((len(x), 1, 1, 1))
    as_t_mul = as_t.detach().clone()
    as_t_mul[is_final > 0.0] = 1.0 
    sigma = torch.sqrt(1. - as_t)
    sigma_cum = (torch.sqrt(1. - a_tminus1))

    coeff = 1.0 / (1.0 + np.exp(-logsnr_max))

    ct_square = sigma_cum / np.sqrt(1.0 - coeff)
    sz_square = (4e-4 * ct_square * (sz_mul ** 2)).reshape((len(x), 1, 1, 1)) * sigma ** 2
    
    for i in range(n_step):
        en = (1.0 + guided_weight) * pi(x, logsnr_tminus1, y=y) - guided_weight * pi(x, logsnr_tminus1, y=y_uncond)
        en = (en - torch.sum((x - xt) ** 2, dim=[1,2,3]) * ct_square * 2e-4 * (1.0 - is_final)) * (sz_mul ** 2)
        grad = torch.autograd.grad(en.sum(), [x])[0]
        x.data = x.data + 0.5 * grad 
        if with_noise:
            x.data = x.data + torch.sqrt(sz_square) * torch.randn_like(x)
        if clip:
            x.data = torch.clamp(x.data, min=-torch.sqrt(as_t_mul), max=torch.sqrt(as_t_mul))

    x.data = x.data / (torch.sqrt(as_t_mul) + 1e-8)
    x.requires_grad = False
    return x.detach()
    
def calculate_fid_with_guide(n_samples, p, pi, real_m, real_s, save_name=None, return_samples=False, guided_weight=None, device=None):
    assert device is not None
    if device == torch.device('cuda', 0):
        print('calculate fid')
    start_time = time.time()
    bs = 100
    clip_Langevin=False
    fid_samples = []
    
    for i in range(n_samples // bs):
        cur_samples = gen_guided_samples2(bs, p, pi, xt=None, guided_weight=guided_weight, clip_langevin=clip_Langevin, device=device)
        fid_samples.append(cur_samples.detach().clone())
        if device == torch.device('cuda', 0):
            print("Generate {} samples, time {:.2f}".format((i+1) * bs, time.time() - start_time))

    fid_samples = torch.cat(fid_samples, dim=0)
    fid_samples = (1.0 + torch.clamp(fid_samples, min=-1.0, max=1.0)) / 2.0
    fid = pfw.fid(fid_samples, real_m=real_m, real_s=real_s, device=device)
    if save_name is not None:
        save_images = fid_samples[:100].clone().detach().cpu()
        torchvision.utils.save_image(save_images, save_name, normalize=True, nrow=10)
    if not return_samples:
        return fid
    else:
        return fid, fid_samples        
            
########################## training loop ######################################
def train(args):
    random.seed(seed + args.local_rank)
    np.random.seed(seed + args.local_rank)
    torch.manual_seed(seed + args.local_rank)
    torch.cuda.manual_seed_all(seed + args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', args.local_rank) 
    
    timestamp = str(dt.datetime.now())[:19]
    timestamp = re.sub(r'[\:-]','', timestamp) # replace unwanted chars 
    timestamp = re.sub(r'[\s]','_', timestamp) # with regex and re.sub
    
    img_dir = os.path.join(log_path, timestamp, 'imgs')
    ckpt_dir = os.path.join(log_path, timestamp, 'ckpt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(log_path, timestamp, __file__))
    
    transform_train = transforms.Compose([
        transforms.Resize(im_sz),
        transforms.CenterCrop(im_sz),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(im_sz),
        transforms.CenterCrop(im_sz),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # define dataset
    if dataset_type == 'Imagenet':
        from ImageNet_dataset import ImageNetKaggle
        trainset = ImageNetKaggle('data/i64', "train_64", transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True, drop_last=True)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=train_sampler)
        train_iter = iter(trainloader)
        testset = ImageNetKaggle('data/i64', "val_64", transform_test)
        testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    else:
        raise NotImplementedError

    start_time = time.time()
    print("Begin calculating real image statistics")
    
    
    if stats_path is not None and os.path.exists(stats_path):
        print('Load from pre-calculated files')
        fid_data_true, testset, testloader = None, None, None
        stats = np.load(stats_path)
        real_m, real_s = stats['mu'], stats['sigma']
        fid_data_true, testset, testloader = None, None, None
    else:
        fid_data_true = []
        for x, _ in testloader:
            fid_data_true.append(x)
            if n_fid_samples_full > 0 and len(fid_data_true) >= n_fid_samples_full:
                break
        fid_data_true = torch.cat(fid_data_true, dim=0)
        fid_data_true = (fid_data_true + 1.0) / 2.0
        print("Load in data", fid_data_true.shape, fid_data_true.min(), fid_data_true.max())
        real_m, real_s = pfw.get_stats(fid_data_true, device=device)
        print("Finish calculating real image statistics {:.3f}".format(time.time() - start_time))

    if mode == 'train':
        fid_data_true, testset, testloader = None, None, None
   
    start_time = time.time()
    
    # begin training the model 
    if dataset_type == 'cifar10':
        pi = EBM(add_q=add_q, temb_dim=pi_dim, n_blocks=pi_n_blocks, use_sn=with_sn, act_fn=ebm_act, n_class=n_class, p_cond=p_cond)
        p = Unet(tanh_out=False, residual=True, dim=p_dim, n_class=n_class, p_cond=p_cond)
    elif dataset_type == 'Imagenet':
        pi = EBM(add_q=add_q, ch_mult=(1, 2, 2, 2), temb_dim=pi_dim, n_blocks=pi_n_blocks, use_sn=with_sn, act_fn=ebm_act, n_class=n_class, p_cond=p_cond)
        p = Unet(dim_mults=(1, 1, 1), num_res_blocks=2, tanh_out=False, residual=True, dim=p_dim, im_sz=im_sz, n_class=n_class, p_cond=p_cond)
    else:
        raise NotImplementedError

    pi.to(device)
    pi.train()
    p.train()
    p.to(device)
    pi = nn.parallel.DistributedDataParallel(pi, device_ids=[args.local_rank], output_device=args.local_rank)
    p = nn.parallel.DistributedDataParallel(p, device_ids=[args.local_rank], output_device=args.local_rank)

    pi_ema = ExponentialMovingAverage(pi.parameters(), decay=pi_ema_decay)
    p_ema = ExponentialMovingAverage(p.parameters(), decay=p_ema_decay)
    pi_optimizer = optim.Adam(pi.parameters(), lr=pi_lr, betas=(0.9, 0.999))
    p_optimizer = optim.AdamW(p.parameters(), lr=p_lr, betas=(0.9, 0.999), weight_decay=p_weight_decay) # re_initialize optimizer
    p_lr_scheduler = optim.lr_scheduler.LambdaLR(p_optimizer, lr_lambda=lambda x: min(1.0, (x + 500) / float(warmup_steps)), last_epoch=-1, verbose=False)
    pi_lr_scheduler = optim.lr_scheduler.LambdaLR(pi_optimizer, lr_lambda=lambda x: min(1.0, x / float(warmup_steps)), last_epoch=-1, verbose=False)
    fid, fid_best = 10000, 10000

    start_iter = 0
    if load_path is not None:
        print('loading from ', load_path)
        state_dict = torch.load(load_path)
        pi.load_state_dict(state_dict['pi_state_dict'])
        p.load_state_dict(state_dict['p_state_dict'])
        pi_ema.load_state_dict(state_dict['pi_ema_state_dict'])
        p_ema.load_state_dict(state_dict['p_ema_state_dict'])
        pi_optimizer.load_state_dict(state_dict['pi_optimizer'])
        p_optimizer.load_state_dict(state_dict['p_optimizer'])
        p_lr_scheduler.load_state_dict(state_dict['p_lr_scheduler'])
        pi_lr_scheduler.load_state_dict(state_dict['pi_lr_scheduler'])
        start_iter = ckpt_idx

    if mode == 'fid_vs_w':
        pi.eval()
        p.eval()
        ws = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0]
        
        if device == torch.device('cuda', 0):
            print("Calculating fid under different w")
        with pi_ema.average_parameters():
            with p_ema.average_parameters():
                for w in ws:
                    if device == torch.device('cuda', 0):
                        save_name = '{}/fid_samples_w{:.1f}.png'.format(os.path.join(log_path, timestamp), w)
                    else:
                        save_name = None
                    calculate_fid_with_guide(n_fid_samples_full, p=p, pi=pi, real_m=real_m, real_s=real_s, \
                        save_name=save_name, return_samples=True, guided_weight=w, device=device)
                    if device == torch.device('cuda', 0):
                        print("Current w {}: fid {}".format(w, out_fid))
        return
    
    for counter in range(start_iter, iterations):
        p_optimizer.zero_grad()
        pi_optimizer.zero_grad()

        # generate samples from pi and p
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            x, y = next(train_iter)
            train_sampler.set_epoch(counter)
        
        x = x.to(device)
        y = y.to(device)

        y = y + 1
        y_mask = torch.rand((len(y),)).to(y.device)
        y[y_mask < p_cond] = 0
        t = torch.randint(low=1, high=n_interval + 1, size=(len(x),), dtype=torch.float).to(x.device) / n_interval
        is_final = (t == 1.0).type(torch.float) 
        t_minus1 = t - 1.0 / n_interval
        assert torch.all(t_minus1 >= 0.0) 
        logsnr_t = logsnr_schedule_fn(t, logsnr_max=logsnr_max, logsnr_min=logsnr_min)
        logsnr_tminus1 = logsnr_schedule_fn(t_minus1, logsnr_max=logsnr_max, logsnr_min=logsnr_min)
        if variance_reduction:
            zt_dist = diffusion_forward(x, logsnr=logsnr_t.reshape(len(x), 1, 1, 1))
            ztminus1_dist = diffusion_forward(x, logsnr=logsnr_tminus1.reshape(len(x), 1, 1, 1))
            tmpe = torch.randn_like(x)
            xt = zt_dist['mean'] + zt_dist['std'] * tmpe
            xtminus1 = ztminus1_dist['mean'] + ztminus1_dist['std'] * tmpe
        else:
            zt_dist = diffusion_forward(x, logsnr=logsnr_t.reshape(len(x), 1, 1, 1))
            xt = zt_dist['mean'] + zt_dist['std'] * torch.randn_like(x)
            xtminus1 = denoise_true(xt, x, logsnr_t, logsnr_tminus1)

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
        # update pi
        pi_loss_weight = 1.0 / (sigma_t / sigma_1)
        pi.train()
        pos_energy =  pi(xtminus1 *  torch.sqrt(as_t_mul.reshape((len(x), 1, 1, 1))), logsnr_tminus1, y=y) 
        pos_loss = -(pos_energy * pi_loss_weight).mean()
        pos_loss.backward()
        p.train()
        
        if p_with_noise:
            xtminus1_neg0 = p(xt, logsnr_t, y) + torch.sqrt(beta_t_tilt * as_t_mul).reshape((len(xt), 1, 1, 1)) * torch.randn_like(xt)
        else:
            xtminus1_neg0 = p(xt, logsnr_t, y)

        xtminus1_negk = xtminus1_neg0.detach().clone()
        xtminus1_negk.requires_grad = True
        pi.eval()
        
        xtminus1_negk = Langevin(pi, xtminus1_negk, xt, logsnr_t, logsnr_tminus1, is_final, y=y)
        
        neg_energy = pi(xtminus1_negk * torch.sqrt(as_t_mul.reshape((len(x), 1, 1, 1))), logsnr_tminus1, y=y)
        neg_loss = (neg_energy * pi_loss_weight).mean()
        neg_loss.backward()
        torch.nn.utils.clip_grad_norm(pi.parameters(), max_norm=grad_clip)
        pi_optimizer.step()
        pi_lr_scheduler.step()
        pi_ema.update()

        p_loss_weight = 1.0

        p_loss = torch.sum((xtminus1_negk * torch.sqrt(as_t_mul.reshape((len(x), 1, 1, 1))) - xtminus1_neg0) ** 2, dim=[1,2,3])
        p_loss = (p_loss * p_loss_weight).mean()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm(p.parameters(), max_norm=grad_clip)
        p_optimizer.step()
        p_lr_scheduler.step()
        p_ema.update()
        
        if counter % print_iter == 0 and device == torch.device('cuda', 0):
            
            print("Iter {} time{:.2f} pos en {:.2f} neg en {:.2f} pi loss {:.2f} p loss {:.2f} p lr {} pi lr {}"\
            .format(counter, time.time() - start_time, -pos_loss.item(), neg_loss.item(), pos_loss.item() + neg_loss.item(), p_loss.item(), p_lr_scheduler.get_last_lr()[0], pi_lr_scheduler.get_last_lr()[0]))

        if counter % plot_iter == 0:
            pi.eval()
            p.eval()
            samples = gen_samples(bs=64, p=p, pi=pi, device=device)
            with pi_ema.average_parameters():
                with p_ema.average_parameters():
                    ema_samples = gen_samples(bs=64, p=p, pi=pi, device=device)
                    ema_zeroguided = gen_guided_samples(bs=min(n_class*10, 100), p=p, pi=pi, guided_weight=0.0, device=device)
                    ema_guided = gen_guided_samples(bs=min(n_class*10, 100), p=p, pi=pi, guided_weight=eval_weight, device=device)

            if device == torch.device('cuda', 0):
                save_images = samples.detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_p_samples.png'.format(img_dir, counter), normalize=True, nrow=8)
                save_images = ema_samples.detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_ema_samples.png'.format(img_dir, counter), normalize=True, nrow=8)
                save_images = ema_zeroguided.detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_ema_zero_guided.png'.format(img_dir, counter), normalize=True, nrow=10)
                save_images = ema_guided.detach().cpu()
                torchvision.utils.save_image(torch.clamp(save_images, min=-1.0, max=1.0), '{}/{}_ema_guided.png'.format(img_dir, counter), normalize=True, nrow=10)
            
            pi.train()
            p.train()
        
        if counter > -1 and counter % fid_iter == 0:
            fid_s_time = time.time()
            pi.eval()
            p.eval()
            save_name = None
            if device == torch.device('cuda', 0):
                save_name = '{}/fid_samples_{}.png'.format(img_dir, counter)
            with pi_ema.average_parameters():
                with p_ema.average_parameters():
                    fid = calculate_fid(n_fid_samples, p, pi, real_m, real_s, save_name=save_name, device=device)
            if device == torch.device('cuda', 0) and fid < fid_best:
                fid_best = fid
                print('Saving best', fid)
                save_dict = {
                'pi_state_dict': pi.state_dict(),
                'p_state_dict': p.state_dict(),
                'pi_ema_state_dict': pi_ema.state_dict(),
                'p_ema_state_dict': p_ema.state_dict(),
                'pi_optimizer': pi_optimizer.state_dict(),
                'p_optimizer': p_optimizer.state_dict(),
                'p_lr_scheduler': p_lr_scheduler.state_dict(),
                'pi_lr_scheduler': pi_lr_scheduler.state_dict()
                }

                torch.save(save_dict, os.path.join(ckpt_dir, 'best.pth.tar'))

            if device == torch.device('cuda', 0):
                print("Finish calculating fid time {:.3f} fid {:.3f} / {:.3f}".format(time.time() - fid_s_time, fid, fid_best))
            pi.train()
            p.train() 

        if counter > -1 and (counter % ckpt_iter == 0) and device == torch.device('cuda', 0):
            print('Saving checkpoint')
            save_dict = {
                'pi_state_dict': pi.state_dict(),
                'p_state_dict': p.state_dict(),
                'pi_ema_state_dict': pi_ema.state_dict(),
                'p_ema_state_dict': p_ema.state_dict(),
                'pi_optimizer': pi_optimizer.state_dict(),
                'p_optimizer': p_optimizer.state_dict(),
                'p_lr_scheduler': p_lr_scheduler.state_dict(),
                'pi_lr_scheduler': pi_lr_scheduler.state_dict()
            }

            torch.save(save_dict, os.path.join(ckpt_dir, '{}.pth.tar'.format(counter)))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coop-diffusion')
    parser.add_argument('--local_rank', type=int, metavar='N', help='local')
    train(parser.parse_args())
