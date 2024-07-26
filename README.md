# Official Implementation of "AdaPI: Facilitating DNN Model Adaptivity for Efficient Private Inference in Edge Computing"


Please cite our paper if you use the code ✔
```
@article{zhou2024adapi,
  title={AdaPI: Facilitating DNN Model Adaptivity for Efficient Private Inference in Edge Computing},
  author={Zhou, Tong and Zhao, Jiahui and Luo, Yukui and Xie, Xi and Wen, Wujie and Ding, Caiwen and Xu, Xiaolin},
  booktitle={2024 IEEE/ACM International Conference on Computer Aided Design (ICCAD)},
  year={2024},
  organization={IEEE}
}
```

# General-L0-Pruning
General L0 Pruning for Weight pruning, non-linear pruning, and activation pruning


Some steps to setup the environment and download dataset
```bash
# Create a environment
conda create --name torch_env
#or
conda create --prefix=${HOME}/.conda/envs/torch_env python=3.9
# Then activate the environment
conda activate torch_env
# Install pytorch package
conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
# Install tensorboard to record accuracy/loss stuffs
conda install -c conda-forge tensorboardx
pip install tqdm pytorch_warmup
pip install scipy
```

Download Tiny-ImageNet dataset:
```bash
bash dataset_download/download_tinyimagenet.sh
```

Project location: 
```bash
/home/jiz22029/mpc_proj/General-L0-Pruning
```

## 1. Train a baseline models
Ways to repeat the pretrained model experiment:
```bash
bash scripts/scripts_baseline.sh
```
- You should specify "```--act_type nn.ReLU```" to run the baseline model by using ReLU non-linear function. 
- You can speicify which gpu you will be used by changing "```--gpu 0```". In the scripts, "```nohup python > out.log```" put the execution of python program into background, and direct the command line output to out.log. <br /> 
- You may need to change the dataset path for Tiny-ImageNet dataset through ```--dataset tiny_imagenet --data_path "/gpfs/sharedfs1/cad15002/dataset/tiny-imagenet-200"```
- Model and logging path can be found in: ```./train_cifar/resnet18_in__tiny_imagenet/epoch_400```

- The log/model save path follows ```./train_cifar/{model}__{dataset}/epoch_{epoch}``` way, where ```model``` and ```dataset``` and ```epoch``` donate model, dataset, and epochs uses for training. 
 

## 2. Run Weight + ReLU pruning:

Here are the steps to run the pruning for resnet18_in on CIFAR-10 dataset: 
```bash
bash scripts/scripts_resnet18_c10_train_finetune.sh
```
The first ``` python xxx```  is used for mask training, and second ``` python xxx```  is used for finetuning. 

For each stage, you should uncomment the current ``` python xxx``` which shall be used, and comment the other ``` python xxx``` which is not used. 

The finetuning stage will use the model with weight and mask generated from mask training stage. 

- The L0 based sparsification algorithm concept can be found in [AutoPrune](https://proceedings.neurips.cc/paper_files/paper/2019/file/4efc9e02abdab6b6166251918570a307-Paper.pdf). 

- ```--distil --teacher_arch resnet18_in --teacher_path ./train_cifar/resnet18_in__cifar10/epoch_400/best.pth.tar\``` loaded the resnet18_in baseline as a teacher to conduct distilltion. The distillation used here has KL divergence distillation, and a so called "attention transfer" distillation, which compares the feature map difference for penalty. The original paper can be found it in [Paying More Attention to Attention](https://arxiv.org/abs/1612.03928).

- The current implementation of attention transfer distillation utilizes feature map difference between teacher and student model after the non-linear function (ReLU or f(x) = x), you might need to the position you want to conduct distillation. 

- ```export lamda_wgt=13.0``` gives the penalty for weight density, ```export lamda_relu=65.0``` gives the penalty for ReLU density. You can increase the penalty value if you want to have a higher sparsity. 

- Current code has weight sparsity and ReLU sparsity. You may modify the code for activation sparsity. A easy way to do it is to multiple a binary mask with output of each ```nn.linear``` and ```nn.conv2D``` layer, you can refer to weight sparsity code in the code and add it. 






