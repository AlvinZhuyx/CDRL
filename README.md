# Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood

<p align="center">
<img src=Images/CDRL_training.png />
<img src=Images/CDRL_sampling.png />
</p>
This is the official code implementation for ICLR 2024 spotlight paper [Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood]([https://arxiv.org/abs/2112.10752](https://openreview.net/pdf?id=AyzkDpuqcl) by Yaxuan Zhu, Jianwen Xie, Ying Nian Wu and Ruiqi Gao. 

## Unconditional Generation

<p align="center">
<img src=Images/fid_cifar10.png />
</p>

To train unconditional model on Cifar-10, please use the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=23455 main_uncond.py
```

## Conditional Generation

<p align="center">
<img src=Images/i32_cfg2.png />
</p>

To train conditional model on ImageNet32, please download the [ImageNet](https://image-net.org/download-images) dataset. [Here](https://drive.google.com/file/d/11KGjj3YL8jDu5C4BiPXREjDJAfyBpYzf/view?usp=sharing), we provided a downsampled version of ImageNet in resolution 64 x 64. Please download the data and put it under the data folder. 

Then please use the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=23455 main_cond.py
```


