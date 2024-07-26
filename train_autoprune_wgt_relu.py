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
    if config.distil:
        config_teacher = copy.deepcopy(config)
        config_teacher.multiple_mask = False
        config_teacher.train_mask = False
        config_teacher.arch = config_teacher.teacher_arch
        config_teacher.act_type = 'ReLU_masked_distil'
        config_teacher.weight_prune = False
        config_teacher.relu_prune = True
        teacher_model = model_spWGT_spReLU(config_teacher)
    model.criterion = criterion
    # if 'vgg' in config.arch:
    #     model = vgg.__dict__[config.arch](config, criterion, config.act_type, config.pool_type)
    # elif 'resnet' in config.arch:
    #     model = resnet.__dict__[config.arch](config, criterion, config.act_type, config.pool_type, num_classes = config.num_classes)
    # elif config.arch == "mobilenet_v2":
    #     model = mbnet.__dict__[config.arch](config, criterion, config.act_type, num_classes = config.num_classes)
    

    # config.pretrained_path = '/data3/hop20001/mpc_proj/PASNET_cifar10_search/pretrained/cifar10_baseline/vgg_checkpoint_best.tar'
    # config.pretrained_path = '/data3/hop20001/mpc_proj/PASNET_cifar10_search/pretrained/cifar10_baseline/resnet18.pt'
    ### finetune from checkpoint
    if config.pretrained_path:
        print("==> Load pretrained")
        model.load_pretrained(pretrained_path = config.pretrained_path)
    if config.distil and (config.dataset != 'imagenet'):
        teacher_model.load_pretrained(pretrained_path = config.teacher_path)
    if(config.checkpoint_path):
        config.start_epoch, best_top1 = model.load_check_point(check_point_path = config.checkpoint_path)
    
    model = model.to(device)
    if config.distil:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        criterion_kd = SoftTarget(4.0).to(device)

    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'test')
    pin_memory = (config.dataset == "imagenet")

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                              num_workers=config.workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=config.batch_size,
                             num_workers=config.workers, pin_memory=pin_memory)

    ########### Original evaluation ###########
    # if config.evaluate:
    #     # # ----------------------------
    #     if config.evaluate != 'None':
    #         model.load_pretrained(pretrained_path = config.evaluate)
    #         # checkpoint = torch.load(config.evaluate, map_location="cpu")
    #         # model.load_state_dict(checkpoint['state_dict'])
    #     # print(model)
    #     from torchinfo import summary
    #     device = "cuda"
    #     summary(model, input_size=(1, 3, 32, 32), device=device, depth=3, verbose=2,
    #         col_names=["input_size",
    #                     "output_size",
    #                     "kernel_size"],
    #     )
        
    #     if config.precision == 'full':
    #         validate(val_loader, model, 0, len(val_loader), device, config, logger, writer)
    #     else:
    #         validate_fp16(val_loader, model, 0, len(val_loader), device, config, logger, writer)
    #     model.update_sparse_list()
    #     model.print_sparse_list(logger)
    #     # model.print_alphas(logger)
    #     return
    ########### Original evaluation ###########

        # # ----------------------------
    # weights optimizer
    # w_optim = torch.optim.SGD(model.weights(), config.w_mask_lr, momentum=config.w_momentum,
    #                           weight_decay=config.w_weight_decay)
    # #w_optim = torch.optim.Adam(model.weights(), config.w_mask_lr)
    # # alphas optimizer
    # if config.act_type != 'nn.ReLU':
    #     alpha_optim = torch.optim.Adam(model.alpha_aux(), config.alpha_lr, betas=(0.5, 0.999),
    #                                 weight_decay=config.alpha_weight_decay)

########## Optimizer Parameter setting 1 ##########
    para_setting = [
    {'params': model.weights(), 'lr': config.w_lr, "betas": (0.9, 0.999), "weight_decay": config.w_weight_decay},
    {'params': model.aux(), 'lr': config.w_mask_lr, "betas": (0.9, 0.999)}
]
    w_optim = torch.optim.AdamW(para_setting)
########## Optimizer Parameter setting 2 ##########
#     para_setting = [
#     {'params': model.weights(), 'lr': config.w_lr},
#     {'params': model.aux(), 'lr': config.w_mask_lr}
# ]
#     w_optim = torch.optim.Adam(para_setting)

    # w_optim = torch.optim.Adam(model.weights_and_alpha(), config.w_mask_lr)
    # w_optim = torch.optim.Adam(para_setting)
    

    if config.enable_lookahead: 
        w_optim = Lookahead(w_optim)

    # w_optim = torch.optim.Adam(model.weights_and_alpha(), config.w_mask_lr, weight_decay=config.w_weight_decay)
    # w_optim = torch.optim.SGD(model.weights_and_alpha(), config.w_mask_lr)
    # w_optim = torch.optim.SGD(model.weights_and_alpha(), config.w_mask_lr, momentum=config.w_momentum,
    #                           weight_decay=config.w_weight_decay)


    # param_groups = [
    #     {'optimizer':w_optim,'T_max':config.mask_epochs, 'eta_min':config.w_lr_min},
    #     {'optimizer':alpha_optim,'T_max':config.mask_epochs}
    # ]
 
    lr_scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = w_optim, T_max = config.mask_epochs, 
                        eta_min = config.w_lr_min)
    
    # if config.act_type != 'nn.ReLU':
    #     lr_scheduler_alpha = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = alpha_optim, T_max = config.mask_epochs)        
    # warmup_scheduler = warmup.UntunedLinearWarmup(alpha_optim)

    #### Freeze batch normalization ####
    # model.train_fz_bn(freeze_bn=True)
    model.train_fz_bn(freeze_bn=False)
    # model.change_mask_dropout_ratio(config.mask_dropout)
    # lambda_l1 = 1e-6
    # lambda_l2 = 5e-4

    # training loop
    best_top1 = 0.

    for epoch in range(config.start_epoch, config.mask_epochs):
        if config.train_mask:
            if config.weight_prune:
                model.update_wgt_sparse_list()
                model.print_wgt_sparse_list(logger)
            if config.relu_prune:
                model.update_relu_sparse_list()
                model.print_relu_sparse_list(logger)
        if config.precision == 'full':
            if(config.distil):
                top1_train = train_mask_distil(train_loader, model, w_optim, teacher_model, criterion_kd, epoch, 
                                                 device, config, logger, writer)
            else:
                top1_train = train_mask(train_loader, model, w_optim, epoch, 
                                                 device, config, logger, writer)
        else:
            if(config.distil):
                top1_train = train_mask_distil_fp16(train_loader, model, w_optim, teacher_model, criterion_kd, epoch, 
                                                 device, config, logger, writer)
            else:
                top1_train = train_mask_fp16(train_loader, model, w_optim, epoch, 
                                                 device, config, logger, writer)
        # training without mask update

        # adjust_learning_rate(w_optim, epoch, config)

        ########## learning rate scheduler ##########
        # lr_scheduler_w.step()
        ########## learning rate scheduler ##########

        # lr_scheduler_alpha.step()
        # validation
        cur_step = (epoch+1) * len(train_loader)
        if config.precision == 'full':
            top1 = validate(val_loader, model, epoch, cur_step, device, config, logger, writer)
        else:
            top1 = validate_fp16(val_loader, model, epoch, cur_step, device, config, logger, writer)

        # # save 
        # # save 
        # relu_count = global_density * total_mask/1000
        # # if (global_density - config.ReLU_count*1000/total_mask) < 0.01:
        # if (relu_count - config.ReLU_count) < 2:
        #     if best_top1 < top1:
        #         best_top1 = top1
        #         # best_genotype = genotype
        #         is_best = True
        #     else:
        #         is_best = False
        #     # utils.save_checkpoint(model, config.path, is_best)
        #     # if is_best:
        #     #     save_path = os.path.join(config.path, 'best_mask_train.pth.tar')
        #     # else:
        #     #     save_path = os.path.join(config.path, 'checkpoint_mask_train.pth.tar')
        #     # model.save_checkpoint(epoch, best_top1, is_best, filename=save_path)
        #     if is_best:
        #         save_path = os.path.join(config.path, 'best_mask_train.pth.tar')
        #         model.save_checkpoint(epoch, best_top1, is_best, filename=save_path)
            
        #     logger.info("Current mask training best Prec@1 = {:.4%}".format(best_top1))
        save_path = os.path.join(config.path, 'checkpoint_mask_train.pth.tar')
        model.save_checkpoint(epoch, top1, False, filename=save_path)

    ############ Start to finetune ############
    #### Fixed mask ####
    model.fix_mask()
    # para_group = [{'params': model.weights(), 'initial_lr': config.w_lr}]
    ### Example setting: w_lr = 1e-3, w_weight_decay = 5e-4, momentum = 0.9
    w_optim = torch.optim.SGD(model.weights(), config.w_lr_finetune, momentum=0.9,
                              weight_decay=5e-4) 

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    
    # model.train_fz_bn(freeze_bn=True)
    model.train_fz_bn(freeze_bn=False)

    best_top1 = 0.

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


     
    for epoch in range(config.start_epoch, config.epochs):
        
        if config.precision == 'full':
            if(config.distil):
                top1_train = train_distil(train_loader, model, w_optim, teacher_model, criterion_kd, epoch, 
                                                 device, config, logger, writer)
            else:
                top1_train = train(train_loader, model, w_optim, epoch, 
                                                 device, config, logger, writer)
        else:
            if(config.distil):
                top1_train = train_distil_fp16(train_loader, model, w_optim, teacher_model, criterion_kd, epoch, 
                                                 device, config, logger, writer)
            else:
                top1_train = train_fp16(train_loader, model, w_optim, epoch, 
                                                 device, config, logger, writer)
        # adjust_learning_rate(w_optim, epoch, config)
        if config.optim == 'cos_modified':
            cos_modified_learning_rate(w_optim, epoch, config)
        else:
            lr_scheduler.step()
        # validation
        cur_step = (epoch+1) * len(train_loader)
        if config.precision == 'full':
            top1 = validate(val_loader, model, epoch, cur_step, device, config, logger, writer)
        else:
            top1 = validate_fp16(val_loader, model, epoch, cur_step, device, config, logger, writer)

        # save 
        if best_top1 < top1:
            best_top1 = top1
            # best_genotype = genotype
            is_best = True
        else:
            is_best = False
        # utils.save_checkpoint(model, config.path, is_best)
        if is_best:
            save_path = os.path.join(config.path, 'best.pth.tar')
        else:
            save_path = os.path.join(config.path, 'checkpoint.pth.tar')
        model.save_checkpoint(epoch, best_top1, is_best, filename=save_path)
        logger.info("Current best Prec@1 = {:.4%}".format(best_top1))

        # if (epoch % 20) == 0:
        #     logger.info("Perform validation on training dataset. ")
        #     if config.precision == 'full':
        #         top1_train_wo_dropout = validate_train(train_loader, model, epoch, cur_step, device, config, logger, writer)
        #     else:
        #         top1_train_wo_dropout = validate_train_fp16(train_loader, model, epoch, cur_step, device, config, logger, writer)
        #     logger.info("Final train Prec@1 = {:.4%}".format(top1_train_wo_dropout))

    ########### Later evaluation ###########
    if config.evaluate:

        # from torchinfo import summary
        device = "cuda"
        # summary(model, input_size=(1, 3, 32, 32), device=device, depth=3, verbose=2,
        #     col_names=["input_size",
        #                 "output_size",
        #                 "kernel_size"],
        # )
        
        print('First layer max', model._weights[0][1].max())
        print('First layer min', model._weights[0][1].min())
        if config.precision == 'full':
            validate(val_loader, model, 0, len(val_loader), device, config, logger, writer)
        else:
            validate_fp16(val_loader, model, 0, len(val_loader), device, config, logger, writer)
        # model.print_alphas(logger)
        return
    ########### Later evaluation ###########

    logger.info("Final best validation Prec@1 = {:.4%}".format(best_top1))
    # logger.info("Best Genotype = {}".format(best_genotype))








if __name__ == "__main__":
    main()
