import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.nn.init as init
import types
import logging
from models_util import *
from models_cifar import *
from models_snl import *
import torchvision


### Model with ReLU Replacement(RP)
class model_spWGT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.model = model
        #### Initialize model architecture
        self.config = config
        self.arch = config.arch
        # if config.dataset == "cifar100":
        #     self.model = eval(config.arch + '(num_classes = 100)')
        # elif config.dataset == "cifar10":
        #     self.model = eval(config.arch + '(num_classes = 10)')
        # else:
        #     print("dataset not supported yet")

        if config.dataset != "imagenet":
            self.model = eval(config.arch + '(config)')
            self.model.apply(weights_init)
        else:
            weight = ''
            if config.pretrained:
                if config.arch == 'resnet18':
                    weight = "weights = torchvision.models.ResNet18_Weights.DEFAULT"
                elif config.arch == 'resnet50':
                    weight = "weights = torchvision.models.ResNet50_Weights.DEFAULT"
                elif config.arch == "efficientnet_b0":
                    weight = "weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT"
                elif config.arch == "efficientnet_b2":#torchvision.models.efficientnet_b2
                    weight = "weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT"
                elif config.arch == "regnet_x_1_6gf":
                    weight = "weights = torchvision.models.RegNet_X_1_6GF_Weights.DEFAULT"
                elif config.arch == "regnet_x_800mf":
                    weight = "weights = torchvision.models.RegNet_X_800MF_Weights.DEFAULT"
            self.model = eval("torchvision.models." + config.arch + "({})".format(weight)) 
            # self.model = torchvision.models.regnet_x_1_6gf
            # self.model = eval("torchvision.models." + config.arch + "(pretrained = config.pretrained)") 
            #### Change maxpool to avepool
            if config.arch == 'resnet18':
                self.model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Generating the mask 'm'

        self.layer_pruned = []
        self.eval()
        with torch.no_grad():
            if config.train_mask:
                for layer in self.model.modules():
                    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                        self.layer_pruned.append(layer)
                        layer.weight_aux = nn.Parameter(torch.ones_like(layer.weight))

                        layer.weight.requires_grad = True
                        layer.weight_aux.requires_grad = True
                        nn.init.uniform_(layer.weight_aux)
                    # This is the monkey-patch overriding layer.forward to custom function.
                    # layer.forward will pass nn.Linear with weights: 'w' and 'm' elementwised
                    if isinstance(layer, nn.Linear):
                        layer.forward = types.MethodType(mask_train_forward_linear, layer)

                    if isinstance(layer, nn.Conv2d):
                        layer.forward = types.MethodType(mask_train_forward_conv2d, layer)
            else:
                for layer in self.model.modules():
                    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                        self.layer_pruned.append(layer)
                        layer.weight_aux = nn.Parameter(torch.ones_like(layer.weight))
                        layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                        layer.weight.requires_grad = True
                        layer.weight_aux.requires_grad = False
                        layer.weight_mask.requires_grad = False
                        nn.init.uniform_(layer.weight_aux)
                        layer.weight_mask.data.copy_(STEFunction.apply(layer.weight_aux))
                    # This is the monkey-patch overriding layer.forward to custom function.
                    # layer.forward will pass nn.Linear with weights: 'w' and 'm' elementwised
                    if isinstance(layer, nn.Linear):
                        layer.forward = types.MethodType(mask_fixed_forward_linear, layer)

                    if isinstance(layer, nn.Conv2d):
                        layer.forward = types.MethodType(mask_fixed_forward_conv2d, layer)
    
        ### Initialize _alpha_aux, _weights lists
        ### self._alpha_aux[i] is the ith _alpha_aux parameter
        self._weights_aux = []
        self._weights = []
        self._weights_and_aux = []
        for name, parameter in self.named_parameters():
            if 'weight_aux' in name:
                self._weights_aux.append((name, parameter))    
                self._weights_and_aux.append((name, parameter))             
            else: 
                self._weights.append((name, parameter))
                self._weights_and_aux.append((name, parameter))
            
        # self._alpha_mask = []
        # for name, parameter in self.named_parameters():
        #     if 'alpha_mask' in name:
        #         self._alpha_mask.append((name, parameter))                 
    def weights(self):
        for n, p in self._weights:
            yield p
    def named_weights(self):
        for n, p in self._weights:
            yield n, p
    def weights_and_aux(self):
        for n, p in self._weights_and_aux:
            yield p
    def named_weights_and_aux(self):
        for n, p in self._weights_and_aux:
            yield n, p
    def weights_aux(self):
        for n, p in self._weights_aux:
            yield p
    def named_weights_aux(self):
        for n, p in self._weights_aux:
            yield n, p
    ### Get Total number of gate parameter
    def _get_num_gates(self):
        with torch.no_grad():
            num_gates = torch.tensor(0.)
            for name, weights_aux in self._weights_aux:
                num_gates += weights_aux.numel()
        return num_gates

    def train_fz_bn(self, freeze_bn=True, freeze_bn_affine=True, mode=True):
        """
            Override the default train() to freeze the BN parameters
        """
        # super(VGG, self).train(mode)
        self.train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if (freeze_bn_affine and m.affine == True):
                    m.weight.requires_grad = not freeze_bn
                    m.bias.requires_grad = not freeze_bn

    def load_pretrained(self, pretrained_path = False):
        if pretrained_path:
            if os.path.isfile(pretrained_path):
                print("=> loading checkpoint '{}'".format(pretrained_path))
                checkpoint = torch.load(pretrained_path, map_location = "cpu")   
                # print('state_dict' in checkpoint.keys())  
                if 'state_dict' in checkpoint.keys():
                    pretrained_dict = checkpoint['state_dict']
                else:
                    pretrained_dict = checkpoint

                # pretrained_dict = checkpoint
                model_dict = self.state_dict()

                # print("pretrained_dict", [k for k, v in pretrained_dict.items()])
                # print("model_dict", [k for k, v in model_dict.items()])
                # exit()

                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                
                # print(pretrained_dict)
                # print({k for k, v in pretrained_dict.items()})
                # print({k for k, v in model_dict.items()})
                # exit(0)
                model_dict.update(pretrained_dict) 
                self.load_state_dict(model_dict)
            else:
                print("=> no checkpoint found at '{}'".format(pretrained_path))

    def load_check_point(self, check_point_path = False):
        if os.path.isfile(check_point_path):
            print("=> loading checkpoint from '{}'".format(check_point_path))
            checkpoint = torch.load(check_point_path, map_location = "cpu")
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint at epoch {}"
                  .format(checkpoint['epoch']))
            print("Is best result?: ", checkpoint['is_best'])
            return start_epoch, best_prec1
        else:
            print("=> no checkpoint found at '{}'".format(check_point_path))
               
    def save_checkpoint(self, epoch, best_prec1, is_best, filename='checkpoint.pth.tar'):
        """
        Save the training model
        """
        state = {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'best_prec1': best_prec1,
                'is_best': is_best
            }
        torch.save(state, filename)
    def update_sparse_list(self):
        # Update the sparse_list correspond to alpha_aux
        # Format: [layer name, Total original ReLU count, Pruned count, Pruned percentage]
        # Update the global_sparsity
        # Format: [Total original ReLU count, Pruned count, Pruned percentage]
        if(hasattr(self, "sparse_list")):
            del self.sparse_list
        if(hasattr(self, "global_sparsity")):
            del self.global_sparsity

        self.sparse_list = []
        total_count_global = 0
        sparsity_count_global = 0
        with torch.no_grad():
            for name, param in self._weights_aux:
                weight_mask = STEFunction.apply(param)
                total_count = weight_mask.numel()
                sparsity_count = torch.sum(weight_mask).item()
                sparsity_pert = sparsity_count/total_count
                self.sparse_list.append([name, total_count, sparsity_count, sparsity_pert])
                total_count_global += total_count
                sparsity_count_global += sparsity_count
        sparsity_pert_global = sparsity_count_global/total_count_global
        self.global_sparsity = [total_count_global, sparsity_count_global, sparsity_pert_global]
    def print_sparse_list(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        # Logging alpha data: 
        logger.info("####### ReLU Sparsity #######")
        logger.info("# Layer wise weight sparsity for the model")
        logger.info("# Format: [layer name, Total original ReLU count, remained count, remained percentage]")
        for sparse_list in self.sparse_list:
            logger.info(sparse_list)
        logger.info("#\n Global weight sparsity for the model")
        logger.info("# Format: [Total original ReLU count, remained count, remained percentage]")
        logger.info(self.global_sparsity)
        logger.info("########## End ###########")
                    
        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def fix_mask(self):
        self.eval()
        with torch.no_grad():
            for layer in self.layer_pruned:
                if not hasattr(self, "weight_mask"):
                    layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight_aux.requires_grad = False
                layer.weight_mask.requires_grad = False
                layer.weight_mask.data.copy_(STEFunction.apply(layer.weight_aux))
                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(mask_fixed_forward_linear, layer)

                if isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(mask_fixed_forward_conv2d, layer)

    def mask_fixed_update(self):
        self.eval()
        with torch.no_grad():
            for layer in self.layer_pruned:
                layer.weight_mask.data.copy_(STEFunction.apply(layer.weight_aux))
    
    def mask_density_forward(self):
        total_count_global = 0
        sparse_list = []
        sparse_pert_list = []
        sparsity_count_global = 0
        # for name, param in self._weights_aux:
            # weight_mask = STEFunction.apply(param)
        for layer in self.layer_pruned:
            weight_mask = STEFunction.apply(layer.weight_aux)
            total_count = weight_mask.numel()
            sparsity_count = torch.sum(weight_mask)
            sparsity_pert = sparsity_count/total_count
            sparse_list.append(sparsity_count.item())
            sparse_pert_list.append(sparse_list[-1]/total_count)
            total_count_global += total_count
            sparsity_count_global += sparsity_count
        global_density = sparsity_count_global/total_count_global 
        return global_density, sparse_list, sparse_pert_list, total_count_global
    
    def forward(self, x):
        out = self.model(x)
        return out