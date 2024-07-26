import numpy as np
import torch
import math
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import warnings
from functools import partial
import random
from models_util import *
import os

### ReLU with run time initialization method
# mask1: bitmap, 1 means has ReLU, 0 means direct pass.
# mask2: bitmap, 1 means direct pass, 0 means have ReLU
# a*mask2: passed element
# a*mask1: element need to be ReLU
# ReLU(a*mask1) + a*mask2
class ReLU_masked(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_feature = 0
        self.current_feature = 0
        self.init = 1
        self.relu_aux = []
        self.relu_mask = []
        self.train_mask = 1

        self.act_map = []
    def init_w_aux(self, size):
        setattr(self, "relu_aux_{}".format(self.num_feature), nn.Parameter(torch.Tensor(*size)))
        nn.init.uniform_(getattr(self, "relu_aux_{}".format(self.num_feature)), a = 0, b = 1) # weight init for aux parameter, can be truncated normal
        self.relu_aux.append(getattr(self, "relu_aux_{}".format(self.num_feature)))

    ########## Fix the relu mask
    def fix_mask(self):
        for current_feature in range(self.num_feature):
            relu_mask = nn.Parameter(torch.ones_like(self.relu_aux[current_feature]))
            relu_mask.requires_grad = False
            self.relu_aux[current_feature].requires_grad = False
            relu_mask.data.copy_(STEFunction.apply(self.relu_aux[current_feature]))
            setattr(self, "relu_mask_{}".format(current_feature), relu_mask)
            self.relu_mask.append(getattr(self, "relu_mask_{}".format(current_feature)))
            self.train_mask = 0
    def mask_fixed_update(self, threshold = 0):
        self.eval()
        with torch.no_grad():
            for current_feature in range(self.num_feature):
                self.relu_mask[current_feature].data.copy_((self.relu_aux[current_feature] > threshold).float())
                
                # self.relu_mask[current_feature].data.copy_(STEFunction.apply(self.relu_aux[current_feature]))

    def mask_density_forward(self):
        l0_reg = 0
        sparse_list = []
        sparse_pert_list = []
        total_mask = 0
        for current_feature in range(self.num_feature):
            neuron_mask = STEFunction.apply(getattr(self, "relu_aux_{}".format(current_feature)))
            l0_reg += torch.sum(neuron_mask)
            sparse_list.append(torch.sum(neuron_mask).item())
            sparse_pert_list.append(sparse_list[-1]/neuron_mask.numel())
            total_mask += neuron_mask.numel()
        global_density = l0_reg/total_mask 
        return global_density, sparse_list, sparse_pert_list, total_mask

    def forward(self, x):
        ### Initialize the parameter at the beginning
        if self.init:
            x_size = list(x.size())[1:] ### Ignore batch size dimension
            self.init_w_aux(x_size)
            neuron_relu_mask = STEFunction.apply(getattr(self, "relu_aux_{}".format(self.num_feature))) ### Mask for element which applies ReLU
            self.num_feature += 1
        ### Conduct recurrently inference during normal inference and training
        else:
            if self.train_mask: 
                neuron_relu_mask = STEFunction.apply(getattr(self, "relu_aux_{}".format(self.current_feature))) ### Mask for element which applies ReLU
            else: 
                neuron_relu_mask = self.relu_mask[self.current_feature]
            if self.current_feature == 0:
                self.act_map.clear()
            self.current_feature = (self.current_feature + 1) % self.num_feature
        neuron_pass_mask = 1 - neuron_relu_mask  ### Mask for element which ignore ReLU
        # out = self.act(torch.mul(x, neuron_relu_mask)) + torch.mul(x, neuron_pass_mask)
        out = torch.mul(F.relu(x), neuron_relu_mask) + torch.mul(x, neuron_pass_mask)
        
        ######### Save tensor to path #########
        # if not self.init:
        #     # Define the file path to save the tensor
        #     folder_path = os.getcwd() + "/" + "plot_feature"
        #     if not os.path.exists(folder_path):
        #         # Create the folder
        #         os.makedirs(folder_path)
        #         print(f"Folder created at {folder_path}")
        #     else:
        #         print(f"Folder already exists at {folder_path}")
        #     # file_path = folder_path + "/" + "density_0.4_feature.pt"
        #     # # Save the tensor to the file
        #     # print(f"Save tensor to {file_path}")
        #     # torch.save(out[0], file_path)
        #     file_path = folder_path + "/" + "density_0.05_feature_x.pt"
        #     # Save the tensor to the file
        #     print(f"Save tensor to {file_path}")
        #     torch.save(x[0], file_path)
        #     exit()
        ######### Save tensor to path #########
        
        self.act_map.append(out)
        return out


### ReLU with run time initialization method
# mask1: bitmap, 1 means has ReLU, 0 means direct pass.
# mask2: bitmap, 1 means direct pass, 0 means have ReLU
# a*mask2: passed element
# a*mask1: element need to be ReLU
# ReLU(a*mask1) + a*mask2
class ReLU_masked_trainable_act(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_feature = 0
        self.current_feature = 0
        self.init = 1
        self.relu_aux = []
        self.relu_mask = []
        self.train_mask = 1

        self.act_map = []
    def init_w_aux(self, size):
        setattr(self, "relu_aux_{}".format(self.num_feature), nn.Parameter(torch.Tensor(*size)))
        nn.init.uniform_(getattr(self, "relu_aux_{}".format(self.num_feature)), a = 0, b = 1) # weight init for aux parameter, can be truncated normal
        self.relu_aux.append(getattr(self, "relu_aux_{}".format(self.num_feature)))

    ########## Fix the relu mask
    def fix_mask(self):
        for current_feature in range(self.num_feature):
            relu_mask = nn.Parameter(torch.ones_like(self.relu_aux[current_feature]))
            relu_mask.requires_grad = False
            self.relu_aux[current_feature].requires_grad = False
            relu_mask.data.copy_(STEFunction.apply(self.relu_aux[current_feature]))
            setattr(self, "relu_mask_{}".format(current_feature), relu_mask)
            self.relu_mask.append(getattr(self, "relu_mask_{}".format(current_feature)))
            self.train_mask = 0
    def mask_fixed_update(self):
        self.eval()
        with torch.no_grad():
            for current_feature in range(self.num_feature):
                self.relu_mask[current_feature].data.copy_(STEFunction.apply(self.relu_aux[current_feature]))

    def mask_density_forward(self):
        l0_reg = 0
        sparse_list = []
        sparse_pert_list = []
        total_mask = 0
        for current_feature in range(self.num_feature):
            neuron_mask = STEFunction.apply(getattr(self, "relu_aux_{}".format(current_feature)))
            l0_reg += torch.sum(neuron_mask)
            sparse_list.append(torch.sum(neuron_mask).item())
            sparse_pert_list.append(sparse_list[-1]/neuron_mask.numel())
            total_mask += neuron_mask.numel()
        global_density = l0_reg/total_mask 
        return global_density, sparse_list, sparse_pert_list, total_mask

    def forward(self, x):
        ### Initialize the parameter at the beginning
        if self.init:
            x_size = list(x.size())[1:] ### Ignore batch size dimension
            self.init_w_aux(x_size)
            neuron_relu_mask = STEFunction.apply(getattr(self, "relu_aux_{}".format(self.num_feature))) ### Mask for element which applies ReLU
            self.num_feature += 1
        ### Conduct recurrently inference during normal inference and training
        else:
            if self.train_mask: 
                neuron_relu_mask = STEFunction.apply(getattr(self, "relu_aux_{}".format(self.current_feature))) ### Mask for element which applies ReLU
            else: 
                neuron_relu_mask = self.relu_mask[self.current_feature]
            if self.current_feature == 0:
                self.act_map.clear()
            self.current_feature = (self.current_feature + 1) % self.num_feature
        neuron_pass_mask = 1 - neuron_relu_mask  ### Mask for element which ignore ReLU
        # out = self.act(torch.mul(x, neuron_relu_mask)) + torch.mul(x, neuron_pass_mask)
        out = torch.mul(F.relu(x), neuron_relu_mask) + torch.mul(x, neuron_pass_mask)

        self.act_map.append(out)
        return out

### ReLU with recording activation map result
class ReLU_masked_distil(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_feature = 0
        self.current_feature = 0
        self.init = 1

        self.act_map = []

    def forward(self, x):
        ### Initialize the parameter at the beginning
        if self.init:
            self.num_feature += 1
        ### Conduct recurrently inference during normal inference and training
        else:
            if self.current_feature == 0:
                self.act_map.clear()
            self.current_feature = (self.current_feature + 1) % self.num_feature
        out = F.relu(x)
        self.act_map.append(out)
        return out