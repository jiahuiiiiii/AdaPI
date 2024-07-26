import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import util_func.utils as utils
from models_util import *

def train_mask_distil(train_loader, model, w_optim, teacher_model, criterion_kd, epoch, device, config, logger, writer):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    teacher_model.eval()
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # alpha_optim.zero_grad()
        ### Compute cross entropy loss: ###
        # output = model(input)
        # ce_loss = model.criterion(output, target)
        # global_density, sparse_list, sparse_pert_list, total_mask = model.mask_density_forward()

        #### compute gradient and do SGD step
        # loss = ce_loss + lambda0*global_density 
        # loss.backward()
        # alpha_optim.step()

        
        output = model(input)
        ce_loss = model.criterion(output, target)

        with torch.no_grad():
            outputs_t = teacher_model(input)
        kd_loss = criterion_kd(output, outputs_t)
        tf_at_loss = at_loss_model(model, teacher_model)
        loss = 0.1 * ce_loss + 0.9 * kd_loss + 500 * tf_at_loss

        if config.weight_prune:
            global_density_wgt, sparse_list_wgt, sparse_pert_list_wgt, total_mask_wgt = model.mask_density_forward_wgt()
            loss_reg_wgt = config.lamda_wgt*global_density_wgt 
            loss += loss_reg_wgt
        if config.relu_prune:
            global_density_relu, sparse_list_relu, sparse_pert_list_relu, total_mask_relu = model.mask_density_forward_relu()
            loss_reg_relu = config.lamda_relu*global_density_relu 
            loss += loss_reg_relu
        w_optim.zero_grad()
        loss.backward()
        # gradient clipping
        if config.enable_grad_norm:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.mask_epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
            if config.weight_prune:
                logger.info("Weight layerwise density: " + str(sparse_list_wgt) + "\nWeight layerwise density percentage: "
                            + str([ '%.3f' % elem for elem in sparse_pert_list_wgt]) + "\nWeight Global density: "
                            + str(global_density_wgt.item()))
            if config.relu_prune:
                logger.info("ReLU layerwise density: " + str(sparse_list_relu) + "\nReLU layerwise density percentage: "
                            + str([ '%.3f' % elem for elem in sparse_pert_list_relu]) + "\nReLU Global density: "
                            + str(global_density_relu.item()))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        if config.weight_prune:
            writer.add_scalar('Weight Global density', global_density_wgt.item(), cur_step)
            writer.add_scalar('Weight Compression ratio', 1/(global_density_wgt.item()), cur_step)
        if config.relu_prune:
            writer.add_scalar('ReLU Global density', global_density_relu.item(), cur_step)
            writer.add_scalar('ReLU Compression ratio', 1/(global_density_relu.item()), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg

def train_mask_distil_fp16(train_loader, model, w_optim, teacher_model, criterion_kd, epoch, device, config, logger, writer):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    teacher_model.eval()
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # alpha_optim.zero_grad()
        # ### Compute cross entropy loss: ###
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        #     output = model(input)
        #     ce_loss = model.criterion(output, target)

        #     global_density, sparse_list, sparse_pert_list, total_mask = model.mask_density_forward()

        #     loss = ce_loss + lambda0*global_density 
        # loss.backward()
        # alpha_optim.step()
        # scaler.scale(loss).backward()
        # scaler.step(alpha_optim)
        # scaler.update()

        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(input)
            ce_loss = model.criterion(output, target)
            outputs_t = teacher_model(input)
            with torch.no_grad():
                outputs_t = teacher_model(input)
            kd_loss = criterion_kd(output, outputs_t)
            tf_at_loss = at_loss_model(model, teacher_model)
            loss = 0.1 * ce_loss + 0.9 * kd_loss + 500 * tf_at_loss
            if config.weight_prune:
                global_density_wgt, sparse_list_wgt, sparse_pert_list_wgt, total_mask_wgt = model.mask_density_forward_wgt()
                loss_reg_wgt = config.lamda_wgt*global_density_wgt 
                loss += loss_reg_wgt
            if config.relu_prune:
                global_density_relu, sparse_list_relu, sparse_pert_list_relu, total_mask_relu = model.mask_density_forward_relu()
                loss_reg_relu = config.lamda_relu*global_density_relu 
                loss += loss_reg_relu
        # loss.backward()
        # ce_loss.backward()
        w_optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(w_optim)
        # gradient clipping
        if config.enable_grad_norm:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        scaler.step(w_optim)
        scaler.update()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.mask_epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
            if config.weight_prune:
                logger.info("Weight layerwise density: " + str(sparse_list_wgt) + "\nWeight layerwise density percentage: "
                            + str([ '%.3f' % elem for elem in sparse_pert_list_wgt]) + "\nWeight Global density: "
                            + str(global_density_wgt.item()))
            if config.relu_prune:
                logger.info("ReLU layerwise density: " + str(sparse_list_relu) + "\nReLU layerwise density percentage: "
                            + str([ '%.3f' % elem for elem in sparse_pert_list_relu]) + "\nReLU Global density: "
                            + str(global_density_relu.item()))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        if config.weight_prune:
            writer.add_scalar('Weight Global density', global_density_wgt.item(), cur_step)
            writer.add_scalar('Weight Compression ratio', 1/(global_density_wgt.item()), cur_step)
        if config.relu_prune:
            writer.add_scalar('ReLU Global density', global_density_relu.item(), cur_step)
            writer.add_scalar('ReLU Compression ratio', 1/(global_density_relu.item()), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg

def train_distil(train_loader, model, w_optim, teacher_model, criterion_kd, epoch, device, config, logger, writer):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    teacher_model.eval()
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        output = model(input)
        ce_loss = model.criterion(output, target)
        with torch.no_grad():
            outputs_t = teacher_model(input)
        kd_loss = criterion_kd(output, outputs_t)
        tf_at_loss = at_loss_model(model, teacher_model)
        loss = 0.1 * ce_loss + 0.9 * kd_loss + 500 * tf_at_loss
        w_optim.zero_grad()
        loss.backward()
        # gradient clipping
        if config.enable_grad_norm:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg         

def train_distil_fp16(train_loader, model, w_optim, teacher_model, criterion_kd, epoch, device, config, logger, writer):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    teacher_model.eval()
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # w_optim.zero_grad()
        # output = model(input)
        # loss = model.criterion(output, target)
        # loss.backward()
        # # gradient clipping
        # nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(input)
            ce_loss = model.criterion(output, target)
            with torch.no_grad():
                outputs_t = teacher_model(input)
            kd_loss = criterion_kd(output, outputs_t)
            tf_at_loss = at_loss_model(model, teacher_model)
            loss = 0.1 * ce_loss + 0.9 * kd_loss + 500 * tf_at_loss
        # loss_all.backward()
        # ce_loss.backward()
        w_optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(w_optim)
        # gradient clipping
        if config.enable_grad_norm:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        scaler.step(w_optim)
        scaler.update()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg 

def train_mask(train_loader, model, w_optim, epoch, device, config, logger, writer):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()

    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # alpha_optim.zero_grad()
        ### Compute cross entropy loss: ###
        # output = model(input)
        # ce_loss = model.criterion(output, target)
        # global_density, sparse_list, sparse_pert_list, total_mask = model.mask_density_forward()

        # loss = ce_loss + lambda0*global_density 
        # loss.backward()
        # alpha_optim.step()


        
        output = model(input)
        ce_loss = model.criterion(output, target)
        loss = ce_loss
        if config.weight_prune:
            global_density_wgt, sparse_list_wgt, sparse_pert_list_wgt, total_mask_wgt = model.mask_density_forward_wgt()
            loss_reg_wgt = config.lamda_wgt*global_density_wgt 
            loss += loss_reg_wgt
        if config.relu_prune:
            global_density_relu, sparse_list_relu, sparse_pert_list_relu, total_mask_relu = model.mask_density_forward_relu()
            loss_reg_relu = config.lamda_relu*global_density_relu 
            loss += loss_reg_relu
        w_optim.zero_grad()
        loss.backward()
        # gradient clipping
        if config.enable_grad_norm:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.mask_epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
            if config.weight_prune:
                logger.info("Weight layerwise density: " + str(sparse_list_wgt) + "\nWeight layerwise density percentage: "
                            + str([ '%.3f' % elem for elem in sparse_pert_list_wgt]) + "\nWeight Global density: "
                            + str(global_density_wgt.item()))
            if config.relu_prune:
                logger.info("ReLU layerwise density: " + str(sparse_list_relu) + "\nReLU layerwise density percentage: "
                            + str([ '%.3f' % elem for elem in sparse_pert_list_relu]) + "\nReLU Global density: "
                            + str(global_density_relu.item()))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        if config.weight_prune:
            writer.add_scalar('Weight Global density', global_density_wgt.item(), cur_step)
            writer.add_scalar('Weight Compression ratio', 1/(global_density_wgt.item()), cur_step)
        if config.relu_prune:
            writer.add_scalar('ReLU Global density', global_density_relu.item(), cur_step)
            writer.add_scalar('ReLU Compression ratio', 1/(global_density_relu.item()), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg

def train_mask_fp16(train_loader, model, w_optim, epoch, device, config, logger, writer):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # alpha_optim.zero_grad()
        ### Compute cross entropy loss: ###
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        #     output = model(input)
        #     ce_loss = model.criterion(output, target)
        #     global_density, sparse_list, sparse_pert_list, total_mask = model.mask_density_forward()
        #     loss = ce_loss + lambda0*global_density 
        # # loss.backward()
        # # alpha_optim.step()
        # scaler.scale(loss).backward()
        # scaler.step(alpha_optim)
        # scaler.update()

        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(input)
            ce_loss = model.criterion(output, target)
            loss = ce_loss
            if config.weight_prune:
                global_density_wgt, sparse_list_wgt, sparse_pert_list_wgt, total_mask_wgt = model.mask_density_forward_wgt()
                loss_reg_wgt = config.lamda_wgt*global_density_wgt 
                loss += loss_reg_wgt
            if config.relu_prune:
                global_density_relu, sparse_list_relu, sparse_pert_list_relu, total_mask_relu = model.mask_density_forward_relu()
                loss_reg_relu = config.lamda_relu*global_density_relu 
                loss += loss_reg_relu
        # ce_loss.backward()
        w_optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(w_optim)
        # gradient clipping
        if config.enable_grad_norm:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        scaler.step(w_optim)
        scaler.update()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.mask_epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
            if config.weight_prune:
                logger.info("Weight layerwise density: " + str(sparse_list_wgt) + "\nWeight layerwise density percentage: "
                            + str([ '%.3f' % elem for elem in sparse_pert_list_wgt]) + "\nWeight Global density: "
                            + str(global_density_wgt.item()))
            if config.relu_prune:
                logger.info("ReLU layerwise density: " + str(sparse_list_relu) + "\nReLU layerwise density percentage: "
                            + str([ '%.3f' % elem for elem in sparse_pert_list_relu]) + "\nReLU Global density: "
                            + str(global_density_relu.item()))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        if config.weight_prune:
            writer.add_scalar('Weight Global density', global_density_wgt.item(), cur_step)
            writer.add_scalar('Weight Compression ratio', 1/(global_density_wgt.item()), cur_step)
        if config.relu_prune:
            writer.add_scalar('ReLU Global density', global_density_relu.item(), cur_step)
            writer.add_scalar('ReLU Compression ratio', 1/(global_density_relu.item()), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg

def train(train_loader, model, w_optim, epoch, device, config, logger, writer):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        output = model(input)
        loss = model.criterion(output, target)
        w_optim.zero_grad()
        loss.backward()
        # gradient clipping
        if config.enable_grad_norm:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg         

def train_fp16(train_loader, model, w_optim, epoch, device, config, logger, writer):
    """
        Run one train epoch
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    # switch to train mode
    total_mask = model._get_num_gates().item()
    model.train()
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for step, (input, target) in enumerate(train_loader):
        N = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # w_optim.zero_grad()
        # output = model(input)
        # loss = model.criterion(output, target)
        # loss.backward()
        # # gradient clipping
        # nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = model(input)
            loss = model.criterion(output, target)
        # ce_loss.backward()
        w_optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(w_optim)
        # gradient clipping
        if not config.enable_grad_norm:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        # w_optim.step()
        scaler.step(w_optim)
        scaler.update()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    return top1.avg 