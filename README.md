# Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood

<p align="center">
<img src=Images/CDRL_training.png />
<img src=Images/CDRL_sampling.png />
</p>

This is the official code implementation for ICLR 2024 spotlight paper [Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood](https://openreview.net/pdf?id=AyzkDpuqcl) by Yaxuan Zhu, Jianwen Xie, Ying Nian Wu and Ruiqi Gao. 

## Unconditional Generation

<p align="center">
<img src=Images/fid_cifar10.png />
</p>

We provide our pretrained ckpt in this [link](https://drive.google.com/file/d/1DAH5V3aoRlCYSp8FAzmWFP7ztb01urED/view?usp=drive_link). To perform test using this ckpt, please change the [code](main_uncond.py) according to the following example:

```
mode = 'fid' # 'train', 'fid'
ckpt_idx = 400000
load_dir = YOUR_CKPT_PATH
load_path = '{}/Cifar_10_best.pth.tar'.format(load_dir)
```

To train unconditional model on Cifar-10, please use the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=23455 main_uncond.py
```

## Conditional Generation

<p align="center">
<img src=Images/i32_cfg2.png />
</p>
We provide the pretrained checkpoint for Imagenet32 [here](https://drive.google.com/file/d/14QoJd_tT1_IaftTjxX4FNyEMT8MOk-Nw/view?usp=drive_link). To run testing on different guidance scale. Please the [code](main_cond.py) according to the following example:

```
mode = 'fid_vs_w' # 'train', 'fid_vs_w'

ckpt_idx = 400000
load_path = 'YOUR_CKPT_PATH/cond_best.pth.tar'
```

To train conditional model on ImageNet32, please download the [ImageNet](https://image-net.org/download-images) dataset. [Here](https://drive.google.com/file/d/11KGjj3YL8jDu5C4BiPXREjDJAfyBpYzf/view?usp=sharing), we provided a downsampled version of ImageNet in resolution 64 x 64 (Given that we only carry out experiments on the resolution of 32 x 32, this downsampled version is sufficient for our usage). Please download the data and extract the contents to data/i64 folder. 

Then please download the label using the following command:
```
cd data/i64
wget https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json
wget https://gist.githubusercontent.com/paulgavrikov/3af1efe6f3dff63f47d48b91bb1bca6b/raw/00bad6903b5e4f84c7796b982b72e2e617e5fde1/ILSVRC2012_val_labels.json
```
We can then run the training use the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=23455 main_cond.py
```


