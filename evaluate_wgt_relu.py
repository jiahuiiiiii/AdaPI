""" Search cell """
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from util_func.config import TrainCifarConfig
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
config = TrainCifarConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.dataset)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    device = torch.device("cuda")
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True
    # model = model_spWGT(config)
    model = model_spWGT_spReLU(config)
    criterion = nn.CrossEntropyLoss().to(device)

    model.criterion = criterion
    ### finetune from checkpoint
    if config.pretrained_path:
        print("==> Load pretrained")
        model.load_pretrained(pretrained_path = config.pretrained_path)
    
    model = model.to(device)

    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'test')
    pin_memory = (config.dataset == "imagenet")

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                              num_workers=config.workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=config.batch_size,
                             num_workers=config.workers, pin_memory=pin_memory)


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
    if config.evaluate:

        # from torchinfo import summary
        device = "cuda"
        
        # print('First layer max', model._weights[0][1].max())
        # print('First layer min', model._weights[0][1].min())
        if config.precision == 'full':
            validate(val_loader, model, 0, len(val_loader), device, config, logger, writer)
        else:
            validate_fp16(val_loader, model, 0, len(val_loader), device, config, logger, writer)
        # model.print_alphas(logger)
        return








if __name__ == "__main__":
    main()
