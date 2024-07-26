""" Search cell """
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from util_func.config import CombineConfig
import util_func.utils as utils
# from models.search_cnn import SearchCNNController
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim
import torch.utils.data
import pytorch_warmup as warmup
from models_util import *
from models_cifar import *
from train_util import *
from util_func.dataset import get_dataset, DATASETS
import math
import copy
config = CombineConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.dataset)))
config.print_params(logger.info)



def save_bn(self, path):
    model_dict = self.state_dict()
    bn_dict = {}
    for k, v in model_dict.items():
        if 'bn' in k:
            bn_dict[k] = v
    # print("model_dict", [k for k, v in model_dict.items()])
    # print("bn_dict", [k for k, v in bn_dict.items()])

    torch.save(bn_dict, path)


def save_wgt(self, path):
    model_dict = self.state_dict()
    wgt_dict = {}
    for k, v in model_dict.items():
        if 'bn' not in k:
            wgt_dict[k] = v
    # print("model_dict", [k for k, v in model_dict.items()])
    # print("wgt_dict", [k for k, v in wgt_dict.items()])
    torch.save(wgt_dict, path)

def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    device = torch.device("cuda")


    torch.backends.cudnn.benchmark = True
    # model = model_spWGT(config)
    model_spWGT_spReLU.save_bn = save_bn
    model_spWGT_spReLU.save_wgt = save_wgt
    model = model_spWGT_spReLU(config)
    criterion = nn.CrossEntropyLoss().to(device)

    model.criterion = criterion
    if config.combine:
        for i in range(4):
            print("==> Load pretrained from:", eval("config.pretrained_path_{}".format(i+1)))
            model.load_pretrained(pretrained_path = eval("config.pretrained_path_{}".format(i+1)))
            
            model.save_bn(os.path.join(config.path, 'bn_{}.pt'.format(i+1)))
            if i == 3:
                model.save_wgt(os.path.join(config.path, 'wgt_0.4.pt'))
    else:


        train_dataset = get_dataset(config, 'train')
        val_dataset = get_dataset(config, 'test')
        pin_memory = (config.dataset == "imagenet")

        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                num_workers=config.workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=config.batch_size,
                                num_workers=config.workers, pin_memory=pin_memory)

        # exit()
        for i in range(4):
            model.load_pretrained(pretrained_path = os.path.join(config.path, 'wgt_0.4.pt'))
            model.load_pretrained(os.path.join(config.path, 'bn_{}.pt'.format(i+1)))

            model = model.to(device)
            config.forward_density = 0.05 * 2**i
            config.forward_density_previous = 0.0001
            print("Density: {}, and {}".format(config.forward_density, config.forward_density_previous))


            ############ Start to finetune ############
            #### Fixed mask ####
            model.fix_mask()
            if config.multiple_mask:
                model.init_multiple_mask(forward_density = config.forward_density, \
                                        forward_density_previous = config.forward_density_previous)
                
            if config.finetune_mask:
                if config.weight_prune:
                    model.update_wgt_sparse_list()
                    model.print_wgt_sparse_list(logger)
                if config.relu_prune:
                    model.update_relu_sparse_list()
                    model.print_relu_sparse_list(logger)
            ########### Later evaluation ###########

            # from torchinfo import summary
            device = "cuda"
            

            config.epochs = 0
            # print('First layer max', model._weights[0][1].max())
            # print('First layer min', model._weights[0][1].min())
            if config.precision == 'full':
                validate(val_loader, model, 0, len(val_loader), device, config, logger, writer)
            else:
                validate_fp16(val_loader, model, 0, len(val_loader), device, config, logger, writer)
            # model.print_alphas(logger)








if __name__ == "__main__":
    main()
