import math
import torchvision
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torchvision import datasets, transforms
import warnings
from functools import partial
import random
from models_util import *

##### Borrowed from https://github.com/szagoruyko/attention-transfer/blob/master/utils.py
def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

def at_loss_model(model_s, model_t):
    loss = 0
    for act_s, act_t in zip(model_s._ReLU_sp_models[0].act_map, model_t._ReLU_sp_models[0].act_map):
        loss += at_loss(act_s, act_t)
    return loss

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss

def replace_relu(model, replacement_fn):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            model._modules[name] = replacement_fn
        else:
            replace_relu(module, replacement_fn)

def replace_siLU(model, replacement_fn):
    for name, module in model.named_children():
        if isinstance(module, nn.SiLU):
            model._modules[name] = replacement_fn
        else:
            replace_siLU(module, replacement_fn)

class STEFunction(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (gate) """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return torch.mul(F.softplus(input), grad_output)

class weight_multi_mask_grad(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (mask) """
    @staticmethod
    def forward(ctx, weight, weight_mask_1, weight_mask_2):
        ctx.save_for_backward(weight_mask_2)
        return torch.mul(weight, weight_mask_1)

    @staticmethod
    def backward(ctx, grad_output):
        weight_mask_2, = ctx.saved_tensors
        return torch.mul(grad_output, weight_mask_2), None, None



class STEFunction_norm(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (gate) """
    @staticmethod
    def forward(ctx, weight_aux, weight):
        weight_mask = (weight_aux > 0).float()
        weight_masked = weight_mask * weight
        ctx.save_for_backward(weight_aux, weight, weight_mask)
        return weight_masked

    @staticmethod
    def backward(ctx, grad_output):
        weight_aux, weight, weight_mask = ctx.saved_tensors
        ######## weight_aux_grad calculation correspond to ########
        ######## Eq. 7 in original AutoPrune paper ########
        weight_aux_grad = torch.mul(weight.sign(), grad_output)
        # weight_inv_abs = torch.reciprocal( torch.abs(weight).clamp(min=1e-4) )
        weight_aux_grad = torch.mul(weight_aux_grad, F.softplus(weight_aux))
        weight_grad = torch.mul(grad_output, weight_mask)
        return weight_aux_grad, weight_grad
    
def mask_train_forward_conv2d_norm(self, x):
    weight_masked = STEFunction_norm.apply(self.weight_aux, self.weight)
    return F.conv2d(x, weight_masked, self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

def mask_train_forward_linear_norm(self, x):
    weight_masked = STEFunction_norm.apply(self.weight_aux, self.weight)
    return F.linear(x, weight_masked, self.bias)

# def mask_fixed_forward_conv2d_norm(self, x):
#     return F.conv2d(x, self.weight * self.weight_mask, self.bias, self.stride, 
#                     self.padding, self.dilation, self.groups)

# def mask_fixed_forward_linear_norm(self, x):
#     return F.linear(x, self.weight * self.weight_mask, self.bias)

def mask_train_forward_conv2d(self, x):
    weight_mask = STEFunction.apply(self.weight_aux)
    return F.conv2d(x, self.weight * weight_mask, self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

def mask_train_forward_linear(self, x):
    weight_mask = STEFunction.apply(self.weight_aux)
    return F.linear(x, self.weight * weight_mask, self.bias)

def mask_fixed_forward_conv2d(self, x):
    # weight_mask = STEFunction.apply(self.weight_aux)
    return F.conv2d(x, self.weight * self.weight_mask, self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

def mask_fixed_forward_linear(self, x):
    # weight_mask = STEFunction.apply(self.weight_aux)
    return F.linear(x, self.weight * self.weight_mask, self.bias)

def multiple_mask_fixed_forward_conv2d(self, x):
    return F.conv2d(x, weight_multi_mask_grad.apply(self.weight, self.weight_mask, self.weight_mask_backward), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)
#weight_multi_mask_grad.apply(self.weight, self.weight_mask, self.weight_mask_backward)
def multiple_mask_fixed_forward_linear(self, x):
    return F.linear(x, weight_multi_mask_grad.apply(self.weight, self.weight_mask, self.weight_mask_backward), self.bias)

def weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def exact_quantile_large_tensor(large_tensor, quantile_value):
    # Flatten and sort the tensor
    sorted_tensor = torch.sort(large_tensor.flatten()).values

    # Calculate the index for the desired quantile
    index = int((large_tensor.numel() - 1) * quantile_value)

    # Get the value at the calculated index
    exact_quantile = sorted_tensor[index]
    return exact_quantile