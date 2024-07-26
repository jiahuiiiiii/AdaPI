import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.nn.init as init
import types
import logging
from models_cifar import *
from models_snl import *
from models_util import *
import torchvision



### Model with ReLU Replacement(RP)
class model_spWGT_spReLU(nn.Module):
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

        # Initialize the weight mask
        self.layer_pruned = []
        self.eval()
        with torch.no_grad():
            if config.weight_prune:
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
        if config.relu_prune:
            # Initialize the activation mask
            self.x_size = config.x_size #### Input image size, for example in cifar 10, it's [1, 3, 32, 32]
            ReLU_masked_model = eval(config.act_type + '(config)') #ReLU_masked()
            # ReLU_masked_model = ReLU_masked(config) #ReLU_masked()
            if "efficientnet" in config.arch:
                replace_siLU(self, ReLU_masked_model)
            else: 
                replace_relu(self, ReLU_masked_model)
            #### Get the name and model_stat of sparse ReLU model ####
            self._ReLU_sp_models = []
            for name, model_stat in self.named_modules(): 
                if 'ReLU_masked' in type(model_stat).__name__:
                    self._ReLU_sp_models.append(model_stat)

            #### Initialize alpha_aux pameters in ReLU_sp model ####
            #### through single step inference ####
            self.eval()
            with torch.no_grad():
                in_mock_tensor = torch.Tensor(*self.x_size)
                self.forward(in_mock_tensor)
                del in_mock_tensor
                for model in self._ReLU_sp_models:
                    model.init = 0
            
        ### Initialize _alpha_aux, _weights lists
        ### self._alpha_aux[i] is the ith _alpha_aux parameter
        self._aux = []
        self._weights = []
        self._weights_aux = []
        self._relu_aux = []
        self._weights_and_aux = []
        for name, parameter in self.named_parameters():
            if 'aux' in name:
                self._aux.append((name, parameter)) 
                if 'weight' in name: 
                    self._weights_aux.append((name, parameter))
                if 'relu' in name: 
                    self._relu_aux.append((name, parameter))
                self._weights_and_aux.append((name, parameter))             
            else: 
                self._weights.append((name, parameter))
                self._weights_and_aux.append((name, parameter))
        self.multiple_mask = False
        # self._alpha_mask = []
        # for name, parameter in self.named_parameters():
        #     if 'alpha_mask' in name:
        #         self._alpha_mask.append((name, parameter))                 
    def init_multiple_mask(self, forward_density = 0.9, forward_density_previous = 0.1):
        self.eval()
        self.multiple_mask = True
        # flatten and concatenate tensors
        all_weights_aux = torch.cat([weight[1].view(-1) for weight in self._weights_aux])
        all_relu_aux = torch.cat([weight[1].view(-1) for weight in self._relu_aux])
        ##### find the percentage value as threshold T
        # print("Weight mask shape:", all_weights_aux.shape)
        # print("ReLU mask shape:", all_relu_aux.shape)
        ##### Small tensor get threshold
        # self.w_aux_forward_threshold = torch.quantile(all_weights_aux, (1 - forward_density))
        # self.w_aux_forward_threshold_previous = torch.quantile(all_weights_aux, (1 - forward_density_previous))
        # self.relu_aux_forward_threshold = torch.quantile(all_relu_aux, (1 - forward_density))
        ##### Large tensor get threshold
        self.w_aux_forward_threshold = exact_quantile_large_tensor(all_weights_aux, (1 - forward_density))
        self.w_aux_forward_threshold_previous = exact_quantile_large_tensor(all_weights_aux, (1 - forward_density_previous))
        self.relu_aux_forward_threshold = exact_quantile_large_tensor(all_relu_aux, (1 - forward_density))


        # self.w_aux_forward_threshold = torch.quantile(all_weights_aux, (1 - 0.064735239))
        # self.w_aux_forward_threshold_previous = torch.quantile(all_weights_aux, (1 - forward_density_previous))
        # self.relu_aux_forward_threshold = torch.quantile(all_relu_aux, (1 - 0.0531738))

        # self.w_aux_forward_threshold = 0
        # self.w_aux_forward_threshold_previous = 0
        # self.relu_aux_forward_threshold = 0

        self.weight_mask = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                # layer.weight_mask_forward = nn.Parameter(torch.ones_like(layer.weight))
                # layer.weight_mask_forward.requires_grad = False
                layer.weight_mask.requires_grad = False
                
                layer.weight_mask_backward = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight_mask_backward.requires_grad = False

                weight_mask = (layer.weight_aux > self.w_aux_forward_threshold).float()
                weight_mask_backward = (torch.logical_and(layer.weight_aux > self.w_aux_forward_threshold, \
                    layer.weight_aux < self.w_aux_forward_threshold_previous)).float()
                layer.weight_mask.data.copy_(weight_mask)
                layer.weight_mask_backward.data.copy_(weight_mask_backward)
                self.weight_mask.append(layer.weight_mask)
            # This is the monkey-patch overriding layer.forward to custom function.
            # layer.forward will pass nn.Linear with weights: 'w' and 'm' element wised
            # The forward and backward will have different mask
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(multiple_mask_fixed_forward_linear, layer)

            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(multiple_mask_fixed_forward_conv2d, layer)
        if self.config.relu_prune:
            for relu_mask_model in self._ReLU_sp_models:
                relu_mask_model.mask_fixed_update(threshold = self.relu_aux_forward_threshold)

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
    def aux(self):
        for n, p in self._aux:
            yield p
    def named_aux(self):
        for n, p in self._aux:
            yield n, p
    ### Get Total number of gate parameter
    def _get_num_gates(self):
        with torch.no_grad():
            num_gates = torch.tensor(0.)
            for name, aux in self._aux:
                num_gates += aux.numel()
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
    def update_wgt_sparse_list(self):
        # Update the sparse_list correspond to alpha_aux
        # Format: [layer name, Total original ReLU count, Pruned count, Pruned percentage]
        # Update the global_sparsity
        # Format: [Total original ReLU count, Pruned count, Pruned percentage]
        if(hasattr(self, "sparse_list_wgt")):
            del self.sparse_list_wgt
        if(hasattr(self, "global_sparsity_wgt")):
            del self.global_sparsity_wgt
        self.sparse_list_wgt = []
        total_count_global = 0
        sparsity_count_global = 0
        with torch.no_grad():
            for i, name_param in enumerate(self._weights_aux):
                if not self.multiple_mask:
                    name, param = name_param #self.multiple_mask
                    weight_mask = STEFunction.apply(param)
                else:
                    name, _ = name_param #self.multiple_mask
                    weight_mask = self.weight_mask[i] 
                total_count = weight_mask.numel()
                sparsity_count = torch.sum(weight_mask).item()
                sparsity_pert = sparsity_count/total_count
                self.sparse_list_wgt.append([name, total_count, sparsity_count, sparsity_pert])
                total_count_global += total_count
                sparsity_count_global += sparsity_count
        sparsity_pert_global = sparsity_count_global/total_count_global
        self.global_sparsity_wgt = [total_count_global, sparsity_count_global, sparsity_pert_global]
    def print_wgt_sparse_list(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        # Logging alpha data: 
        logger.info("####### Weight Sparsity #######")
        logger.info("# Layer wise weight sparsity for the model")
        logger.info("# Format: [layer name, Total original weight count, remained count, remained percentage]")
        for sparse_list in self.sparse_list_wgt:
            logger.info(sparse_list)
        logger.info("#\n Global weight sparsity for the model")
        logger.info("# Format: [Total original weight count, remained count, remained percentage]")
        logger.info(self.global_sparsity_wgt)
        logger.info("########## End ###########")
                    
        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def update_relu_sparse_list(self):
        # Update the sparse_list correspond to alpha_aux
        # Format: [layer name, Total original ReLU count, Pruned count, Pruned percentage]
        # Update the global_sparsity
        # Format: [Total original ReLU count, Pruned count, Pruned percentage]
        if(hasattr(self, "sparse_list_relu")):
            del self.sparse_list_relu
        if(hasattr(self, "global_sparsity_relu")):
            del self.global_sparsity_relu
        self.sparse_list_relu = []
        total_count_global = 0
        sparsity_count_global = 0
        with torch.no_grad():#.relu_mask
            for i, name_param in enumerate(self._relu_aux):
                if not self.multiple_mask:
                    name, param = name_param
                    weight_mask = STEFunction.apply(param)
                else:
                    name, _ = name_param #self.multiple_mask
                    weight_mask = self._ReLU_sp_models[0].relu_mask[i]
                total_count = weight_mask.numel()
                sparsity_count = torch.sum(weight_mask).item()
                sparsity_pert = sparsity_count/total_count
                self.sparse_list_relu.append([name, total_count, sparsity_count, sparsity_pert])
                total_count_global += total_count
                sparsity_count_global += sparsity_count
        sparsity_pert_global = sparsity_count_global/total_count_global
        self.global_sparsity_relu = [total_count_global, sparsity_count_global, sparsity_pert_global]
    def print_relu_sparse_list(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        # Logging alpha data: 
        logger.info("####### ReLU Sparsity #######")
        logger.info("# Layer wise relu sparsity for the model")
        logger.info("# Format: [layer name, Total original ReLU count, remained count, remained percentage]")
        for sparse_list in self.sparse_list_relu:
            logger.info(sparse_list)
        logger.info("#\n Global relu sparsity for the model")
        logger.info("# Format: [Total original ReLU count, remained count, remained percentage]")
        logger.info(self.global_sparsity_relu)
        logger.info("########## End ###########")
                    
        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)


    def fix_mask(self):
        self.eval()
        with torch.no_grad():
            if self.config.weight_prune:
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
                for name, aux in self._aux:
                    aux.requires_grad = False
            if self.config.relu_prune:
                for ReLU_sp_models in self._ReLU_sp_models:
                    ReLU_sp_models.fix_mask()
    def mask_fixed_update(self):
        self.eval()
        with torch.no_grad():
            if self.config.weight_prune:
                for layer in self.layer_pruned:
                    layer.weight_mask.data.copy_(STEFunction.apply(layer.weight_aux))
            if self.config.relu_prune:
                for ReLU_sp_models in self._ReLU_sp_models:
                    ReLU_sp_models.mask_fixed_update()
    def mask_density_forward_wgt(self):
        total_count_global = 0
        sparse_list = []
        sparse_pert_list = []
        sparsity_count_global = 0
        # for name, param in self._aux:
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
    def mask_density_forward_relu(self):
        return self._ReLU_sp_models[0].mask_density_forward()
    
    def forward(self, x):
        out = self.model(x)
        return out