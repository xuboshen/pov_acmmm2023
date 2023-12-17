#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import random
import torch
import copy
from functools import reduce
from torch.nn.utils import clip_grad_norm_
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models.build import MODEL_REGISTRY
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, EpochTimer, multitask_accuracy, decoupleActions
from slowfast.utils.multigrid import MultigridSchedule
from fvcore.common.timer import Timer
logger = logging.get_logger(__name__)
import os
import sys
import datetime

def train_epoch(
    train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg, length, writer=None
):
    # initialize trainmeter
    train_vmeter = TrainMeter(length, cfg)
    train_nmeter = TrainMeter(length, cfg)
    train_vmeter.iter_tic()
    train_nmeter.iter_tic()
    train_meter.iter_tic()
    model.train()
    data_size = len(train_loader)
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)().cuda()
    for cur_iter, (inputs, labels, boxes) in enumerate(train_loader):
        if isinstance(labels, (list, )):
            if cfg.TRAIN.MULTITASK is True:
                poses = labels[0][0].cuda()
                confidence = labels[0][1].cuda()
                labels = [lb.cuda() for lb in labels[1]]
            else:
                labels = [lb.cuda() for lb in labels]
        else:
            labels = labels.cuda()
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / length, cfg)
        optim.set_lr(optimizer, lr)

        train_vmeter.data_toc()
        train_nmeter.data_toc()
        train_meter.data_toc()
        # order: loss3 > loss1 > loss2 > loss4
        loss_list = []
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            if isinstance(cfg.MODEL.NUM_CLASSES, (list, tuple)):
                if cfg.PROMPT.ENABLE:
                    preds = model(inputs, domain_labels=labels[2])
                else:
                    preds = model(inputs)

                loss1 = loss_fun(preds[0], labels[0]) / cfg.TRAIN.GRADIENT_ACCUMULATION
                loss2 = loss_fun(preds[1], labels[1]) / cfg.TRAIN.GRADIENT_ACCUMULATION
                loss = loss1 + loss2
                loss_list += [loss1, loss2]
                if 'loss3' in vars():
                    loss += loss3
                if 'loss4' in vars():
                    loss += loss4
            else:
                if cfg.PROMPT.ENABLE:
                    if cfg.PROMPT.CONTRASTIVE:
                        feature, preds = model(inputs, return_feat=True, domain_labels=labels[1])
                    else:
                        preds = model(inputs, domain_labels=labels[1], bboxes=boxes)
                    if isinstance(labels, list):
                        labels = labels[0]
                else:
                    preds = model(inputs)
                preds = preds[0]
                loss1 = loss_fun(preds, labels) / cfg.TRAIN.GRADIENT_ACCUMULATION
                loss = loss1
                loss_list += [loss1]
                if cfg.PROMPT.CONTRASTIVE:
                    loss2 = losses.get_loss_func("contrastive_loss")(feature, labels) / cfg.TRAIN.GRADIENT_ACCUMULATION
                    loss += loss2
                    loss_list += [loss2]
                if 'loss3' in vars():
                    loss += loss3
                if 'loss4' in vars():
                    loss += loss4
                if 'domain_loss' in vars():
                    loss += domain_loss
            loss_list += [loss, loss, loss]
        # check Nan Loss.
        misc.check_nan_losses(loss)

        scaler.scale(loss).backward()

        # gradient clip:
        if cfg.TRAIN.CLIP_GRADIENT_NORM is not None:
            total_norm = clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRADIENT_NORM)
        # Update the parameters.
        if (cur_iter + 1) % cfg.TRAIN.GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        top1_v_err, top5_v_err = None, None
        top1_n_err, top5_n_err = None, None
        top1_err, top5_err = None, None
        if isinstance(cfg.MODEL.NUM_CLASSES, (list, tuple)):
            # Compute the errors.
            num_topks_v_correct = metrics.topks_correct(preds[0], labels[0], (1, 5))
            num_topks_n_correct = metrics.topks_correct(preds[1], labels[1], (1, 5))
            top1_err, top5_err = multitask_accuracy((preds[0].cpu(), preds[1].cpu()),
                (labels[0].cpu(), labels[1].cpu()),
                topk=(1, 5))
            top1_v_err, top5_v_err = [
                (x / preds[0].size(0)) * 100.0 for x in num_topks_v_correct
            ]
            top1_n_err, top5_n_err = [
                (x / preds[1].size(0)) * 100.0 for x in num_topks_n_correct
            ]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_list = du.all_reduce(loss_list)
                # for i in range(len(loss_list)):
                #     loss_list[i] = du.all_reduce(loss_list[i])
                top1_v_err, top5_v_err = du.all_reduce(
                    [top1_v_err.cuda(), top5_v_err.cuda()]
                )
                top1_n_err, top5_n_err = du.all_reduce(
                    [top1_n_err.cuda(), top5_n_err.cuda()]
                )
                top1_err, top5_err = du.all_reduce(
                    [top1_err.cuda(), top5_err.cuda()]
                )
            # Copy the stats from GPU to CPU (sync point).
            for i in range(len(loss_list)):
                loss_list[i] = loss_list[i].item()
            top1_err, top5_err = (top1_err.item(), top5_err.item(), )
            top1_v_err, top5_v_err = (top1_v_err.item(), top5_v_err.item())
            top1_n_err, top5_n_err = (top1_n_err.item(), top5_n_err.item())
            # Update and log stats.
            if 'loss3' in vars():
                train_meter.update_stats(top1_err, top5_err, loss_list[0], lr, inputs[0][0].size(0) * max( cfg.NUM_GPUS, 1 ),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                train_vmeter.update_stats(top1_v_err, top5_v_err, loss_list[1], lr, inputs[0][0].size(0) * max(cfg.NUM_GPUS, 1), ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                train_nmeter.update_stats(top1_n_err, top5_n_err, loss_list[2], lr, inputs[0][0].size(0) * max(cfg.NUM_GPUS, 1), ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            else:
                train_vmeter.update_stats(top1_v_err, top5_v_err, loss_list[0], lr, inputs[0].size(0) * max(cfg.NUM_GPUS, 1), ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                train_nmeter.update_stats(top1_n_err, top5_n_err, loss_list[1], lr, inputs[0].size(0) * max(cfg.NUM_GPUS, 1), ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                train_meter.update_stats(top1_err, top5_err, loss_list[2], lr, inputs[0].size(0) * max( cfg.NUM_GPUS, 1 ),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars({"Train/verb loss": loss_list[0], "Train/lr": lr, "Train/Top1_verb_acc": top1_v_err, "Train/Top5_verb_acc": top5_v_err,}, global_step=length * cur_epoch + cur_iter,)
                writer.add_scalars({"Train/noun loss": loss_list[1], "Train/lr": lr, "Train/Top1_verb_acc": top1_n_err, "Train/Top5_verb_acc": top5_n_err,}, global_step=length * cur_epoch + cur_iter,)
                writer.add_scalars( { "Train/loss": loss_list[2], "Train/lr": lr, "Train/Top1_acc": top1_err, "Train/Top5_acc": top5_err, }, global_step=length * cur_epoch + cur_iter, )
            
        else:
            # Compute the errors.
            # print(labels)
            decoupled_preds, decoupled_labels = decoupleActions(preds, labels, dataset=cfg.TRAIN.DATASET.lower())
            num_topks_v_correct = metrics.calTopNAccuracy(decoupled_preds[0], decoupled_labels[0], (1, 5))
            num_topks_n_correct = metrics.calTopNAccuracy(decoupled_preds[1], decoupled_labels[1], (1, 5))
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_v_err, top5_v_err = [
                (x / decoupled_preds[0].size(0)) * 100.0 for x in num_topks_v_correct
            ]
            top1_n_err, top5_n_err = [
                (x / decoupled_preds[1].size(0)) * 100.0 for x in num_topks_n_correct
            ]
            top1_err, top5_err = [
                (x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                top1_v_err, top5_v_err = du.all_reduce(
                    [top1_v_err.cuda(), top5_v_err.cuda()]
                )
                top1_n_err, top5_n_err = du.all_reduce(
                    [top1_n_err.cuda(), top5_n_err.cuda()]
                )
                top1_err, top5_err = du.all_reduce(
                    [top1_err.cuda(), top5_err.cuda()]
                )
                loss_list = du.all_reduce(loss_list)
            for i in range(len(loss_list)):
                loss_list[i] = loss_list[i].item()
            # Copy the stats from GPU to CPU (sync point).
            top1_err, top5_err = (top1_err.item(), top5_err.item(), )
            top1_v_err, top5_v_err = (top1_v_err.item(), top5_v_err.item())
            top1_n_err, top5_n_err = (top1_n_err.item(), top5_n_err.item())
            # Update and log stats.
            if 'loss3' in vars():
                train_vmeter.update_stats(top1_v_err, top5_v_err, loss_list[0], lr, inputs[0][0].size(0) * max(cfg.NUM_GPUS, 1), ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                train_nmeter.update_stats(top1_n_err, top5_n_err, loss_list[1], lr, inputs[0][0].size(0) * max(cfg.NUM_GPUS, 1), ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                train_meter.update_stats(top1_err, top5_err, loss_list[2], lr, inputs[0][0].size(0) * max( cfg.NUM_GPUS, 1 ),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            else:
                train_vmeter.update_stats(top1_v_err, top5_v_err, loss_list[0], lr, inputs[0].size(0) * max(cfg.NUM_GPUS, 1), ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                train_nmeter.update_stats(top1_n_err, top5_n_err, loss_list[1], lr, inputs[0].size(0) * max(cfg.NUM_GPUS, 1), ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                train_meter.update_stats(top1_err, top5_err, loss_list[2], lr, inputs[0].size(0) * max( cfg.NUM_GPUS, 1 ),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars({"Train/verb loss": loss_list[0], "Train/lr": lr, "Train/Top1_verb_acc": top1_v_err, "Train/Top5_verb_acc": top5_v_err,}, global_step=length * cur_epoch + cur_iter,)
                writer.add_scalars({"Train/noun loss": loss_list[1], "Train/lr": lr, "Train/Top1_verb_acc": top1_n_err, "Train/Top5_verb_acc": top5_n_err,}, global_step=length * cur_epoch + cur_iter,)
                writer.add_scalars( { "Train/loss": loss_list[2], "Train/lr": lr, "Train/Top1_acc": top1_err, "Train/Top5_acc": top5_err, }, global_step=length * cur_epoch + cur_iter, )
        train_vmeter.iter_toc()  # measure allreduce for this meter
        train_vmeter.log_iter_stats(cur_epoch, cur_iter, "verb")
        train_vmeter.iter_tic()      
        train_nmeter.iter_toc()  # measure allreduce for this meter
        train_nmeter.log_iter_stats(cur_epoch, cur_iter, "noun")
        train_nmeter.iter_tic()      
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        if cfg.DEBUG and cur_iter == 20:
            break
        cur_iter += 1
    # Log epoch stats.
    train_vmeter.log_epoch_stats(cur_epoch, "verb")
    train_vmeter.reset()
    train_nmeter.log_epoch_stats(cur_epoch, "noun")
    train_nmeter.reset()
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch_without(val_loader, model, val_meter, cur_epoch, cfg, length, writer=None):

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_vmeter = ValMeter(length, cfg)
    val_nmeter = ValMeter(length, cfg)
    val_vmeter.iter_tic()
    val_nmeter.iter_tic()
    val_meter.iter_tic()
    data_size = len(val_loader)
    for cur_iter, (inputs, labels, boxes) in enumerate(val_loader):
        if cfg.TRAIN.PARTIAL_DS is not None and cur_iter >= cfg.TRAIN.PARTIAL_DS * data_size:
            break
        if isinstance(inputs, list) and (cfg.MODEL.MODEL_NAME == "TSM" or cfg.MODEL.MODEL_NAME == 'MSG3D'):
            inputs = inputs[0]
        if isinstance(labels, (list, )):
            labels = [lb.cuda() for lb in labels]
        else:
            labels = labels.cuda()
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # for key, val in meta.items():
            #     if isinstance(val, (list,)):
            #         for i in range(len(val)):
            #             val[i] = val[i].cuda(non_blocking=True)
            #     else:
            #         meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()
        val_vmeter.data_toc()
        val_nmeter.data_toc()
        preds = model(inputs)

        if isinstance(cfg.MODEL.NUM_CLASSES, (list, tuple)):
            # Compute the errors.
            num_topks_v_correct = metrics.topks_correct(preds[0], labels[0], (1, 5))
            num_topks_n_correct = metrics.topks_correct(preds[1], labels[1], (1, 5))
            top1_err, top5_err = multitask_accuracy((preds[0].cpu(), preds[1].cpu()),
                (labels[0].cpu(), labels[1].cpu()),
                topk=(1, 5))
            top1_v_err, top5_v_err = [
                (x / preds[0].size(0)) * 100.0 for x in num_topks_v_correct
            ]
            top1_n_err, top5_n_err = [
                (x / preds[1].size(0)) * 100.0 for x in num_topks_n_correct
            ]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                top1_v_err, top5_v_err = du.all_reduce(
                    [top1_v_err.cuda(), top5_v_err.cuda()]
                )
                top1_n_err, top5_n_err = du.all_reduce(
                    [top1_n_err.cuda(), top5_n_err.cuda()]
                )
                top1_err, top5_err = du.all_reduce(
                    [top1_err.cuda(), top5_err.cuda()]
                )     
            # Copy the stats from GPU to CPU (sync point).
            top1_err, top5_err = (top1_err.item(), top5_err.item(), )
            top1_v_err, top5_v_err = (top1_v_err.item(), top5_v_err.item())
            top1_n_err, top5_n_err = (top1_n_err.item(), top5_n_err.item())
            val_vmeter.iter_toc()
            val_nmeter.iter_toc()
            val_meter.iter_toc()
            # Update and log stats.
            val_vmeter.update_stats(top1_v_err, top5_v_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            val_nmeter.update_stats(top1_n_err, top5_n_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            val_meter.update_stats(top1_err, top5_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
           # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_v_acc": top1_v_err, "Val/Top5_v_acc": top5_v_err},
                    global_step=length * cur_epoch + cur_iter,
                )
                writer.add_scalars(
                    {"Val/Top1_n_acc": top1_n_err, "Val/Top5_n_acc": top5_n_err},
                    global_step=length * cur_epoch + cur_iter,
                )
                writer.add_scalars(
                    {"Val/Top1_action_acc": top1_err, "Val/Top5_action_acc": top5_err},
                    global_step=length * cur_epoch + cur_iter,
                )         
        else:
            preds = preds[0]
            decoupled_preds, decoupled_labels = decoupleActions(preds, labels, dataset=cfg.TRAIN.DATASET.lower())
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            num_topks_v_correct = metrics.calTopNAccuracy(decoupled_preds[0], decoupled_labels[0], (1, 5))
            num_topks_n_correct = metrics.calTopNAccuracy(decoupled_preds[1], decoupled_labels[1], (1, 5))
            top1_err, top5_err = [
                (x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            top1_v_err, top5_v_err = [
                (x / decoupled_preds[0].size(0)) * 100.0 for x in num_topks_v_correct
            ]
            top1_n_err, top5_n_err = [
                (x / decoupled_preds[1].size(0)) * 100.0 for x in num_topks_n_correct
            ]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
                top1_v_err, top5_v_err = du.all_reduce([top1_v_err.cuda(), top5_v_err.cuda()])
                top1_n_err, top5_n_err = du.all_reduce([top1_n_err.cuda(), top5_n_err.cuda()])
            # Copy the stats from GPU to CPU (sync point).
            top1_v_err, top5_v_err = (top1_v_err.item(), top5_v_err.item())
            top1_n_err, top5_n_err = (top1_n_err.item(), top5_n_err.item())
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_vmeter.iter_toc()
            val_nmeter.iter_toc()
            val_meter.iter_toc()
            # Update and log stats.
            val_vmeter.update_stats(top1_v_err, top5_v_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            val_nmeter.update_stats(top1_n_err, top5_n_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            val_meter.update_stats(top1_err, top5_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_v_acc": top1_v_err, "Val/Top5_v_acc": top5_v_err},
                    global_step=length * cur_epoch + cur_iter,
                )
                writer.add_scalars(
                    {"Val/Top1_n_acc": top1_n_err, "Val/Top5_n_acc": top5_n_err},
                    global_step=length * cur_epoch + cur_iter,
                )
                writer.add_scalars(
                    {"Val/Top1_action_acc": top1_err, "Val/Top5_action_acc": top5_err},
                    global_step=length * cur_epoch + cur_iter,
                )

        val_nmeter.update_predictions(preds, labels)
        val_vmeter.update_predictions(preds, labels)
        val_meter.update_predictions(preds, labels)
        val_vmeter.log_iter_stats(cur_epoch, cur_iter, "verb")
        val_nmeter.log_iter_stats(cur_epoch, cur_iter, "noun")
        val_meter.log_iter_stats(cur_epoch, cur_iter, "action")
        val_meter.iter_tic()
        val_vmeter.iter_tic()
        val_nmeter.iter_tic()
        # inputs, labels = val_loader.next()
        

    # Log epoch stats.
    val_vmeter.log_epoch_stats(cur_epoch, "verb")
    val_nmeter.log_epoch_stats(cur_epoch, "noun")
    val_meter.log_epoch_stats(cur_epoch, "action")
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            if isinstance(cfg.MODEL.NUM_CLASSES, (list, tuple)):
                all_vpreds = [pred.clone().detach() for pred in val_vmeter.all_vpreds]
                all_vlabels = [label.clone().detach() for label in val_vmeter.all_vlabels]
                all_npreds = [pred.clone().detach() for pred in val_nmeter.all_npreds]
                all_nlabels = [label.clone().detach() for label in val_nmeter.all_nlabels]
                all_preds = [pred.clone().detach() for pred in val_meter.all_npreds]
                all_labels = [label.clone().detach() for label in val_meter.all_nlabels]
                
                if cfg.NUM_GPUS:
                    all_vpreds = [pred.cpu() for pred in all_vpreds]
                    all_vlabels = [label.cpu() for label in all_vlabels]
                    all_npreds = [pred.cpu() for pred in all_npreds]
                    all_nlabels = [label.cpu() for label in all_nlabels]
                writer.plot_eval(
                    preds=all_vpreds, labels=all_vlabels, global_step=cur_epoch
                )
                writer.plot_eval(
                    preds=all_npreds, labels=all_nlabels, global_step=cur_epoch
                )
            else:
                all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
                all_labels = [
                    label.clone().detach() for label in val_meter.all_labels
                ]
                if cfg.NUM_GPUS:
                    all_preds = [pred.cpu() for pred in all_preds]
                    all_labels = [label.cpu() for label in all_labels]
                writer.plot_eval(
                    preds=all_preds, labels=all_labels, global_step=cur_epoch
                )
    val_meter.reset()
    val_vmeter.reset()
    val_nmeter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, length, writer=None):

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_vmeter = ValMeter(length, cfg)
    val_nmeter = ValMeter(length, cfg)
    val_vmeter.iter_tic()
    val_nmeter.iter_tic()
    val_meter.iter_tic()
    data_size = len(val_loader)
    for cur_iter, (inputs, labels, boxes) in enumerate(val_loader):
        if cfg.TRAIN.PARTIAL_DS is not None and cur_iter >= cfg.TRAIN.PARTIAL_DS * data_size:
            break
        if isinstance(inputs, list) and (cfg.MODEL.MODEL_NAME == "TSM" or cfg.MODEL.MODEL_NAME == 'MSG3D'):
            inputs = inputs[0]
        if isinstance(labels, (list, )):
            labels = [lb.cuda() for lb in labels]
        else:
            labels = labels.cuda()
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # for key, val in meta.items():
            #     if isinstance(val, (list,)):
            #         for i in range(len(val)):
            #             val[i] = val[i].cuda(non_blocking=True)
            #     else:
            #         meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()
        val_vmeter.data_toc()
        val_nmeter.data_toc()
        preds = model(inputs, bboxes=boxes)

        if isinstance(cfg.MODEL.NUM_CLASSES, (list, tuple)):
            # Compute the errors.
            num_topks_v_correct = metrics.topks_correct(preds[0], labels[0], (1, 5))
            num_topks_n_correct = metrics.topks_correct(preds[1], labels[1], (1, 5))
            top1_err, top5_err = multitask_accuracy((preds[0].cpu(), preds[1].cpu()),
                (labels[0].cpu(), labels[1].cpu()),
                topk=(1, 5))
            top1_v_err, top5_v_err = [
                (x / preds[0].size(0)) * 100.0 for x in num_topks_v_correct
            ]
            top1_n_err, top5_n_err = [
                (x / preds[1].size(0)) * 100.0 for x in num_topks_n_correct
            ]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                top1_v_err, top5_v_err = du.all_reduce(
                    [top1_v_err.cuda(), top5_v_err.cuda()]
                )
                top1_n_err, top5_n_err = du.all_reduce(
                    [top1_n_err.cuda(), top5_n_err.cuda()]
                )
                top1_err, top5_err = du.all_reduce(
                    [top1_err.cuda(), top5_err.cuda()]
                )     
            # Copy the stats from GPU to CPU (sync point).
            top1_err, top5_err = (top1_err.item(), top5_err.item(), )
            top1_v_err, top5_v_err = (top1_v_err.item(), top5_v_err.item())
            top1_n_err, top5_n_err = (top1_n_err.item(), top5_n_err.item())
            val_vmeter.iter_toc()
            val_nmeter.iter_toc()
            val_meter.iter_toc()
            # Update and log stats.
            val_vmeter.update_stats(top1_v_err, top5_v_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            val_nmeter.update_stats(top1_n_err, top5_n_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            val_meter.update_stats(top1_err, top5_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
           # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_v_acc": top1_v_err, "Val/Top5_v_acc": top5_v_err},
                    global_step=length * cur_epoch + cur_iter,
                )
                writer.add_scalars(
                    {"Val/Top1_n_acc": top1_n_err, "Val/Top5_n_acc": top5_n_err},
                    global_step=length * cur_epoch + cur_iter,
                )
                writer.add_scalars(
                    {"Val/Top1_action_acc": top1_err, "Val/Top5_action_acc": top5_err},
                    global_step=length * cur_epoch + cur_iter,
                )         
        else:
            preds = preds[0]
            decoupled_preds, decoupled_labels = decoupleActions(preds, labels, dataset=cfg.TRAIN.DATASET.lower())
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            num_topks_v_correct = metrics.calTopNAccuracy(decoupled_preds[0], decoupled_labels[0], (1, 5))
            num_topks_n_correct = metrics.calTopNAccuracy(decoupled_preds[1], decoupled_labels[1], (1, 5))
            top1_err, top5_err = [
                (x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            top1_v_err, top5_v_err = [
                (x / decoupled_preds[0].size(0)) * 100.0 for x in num_topks_v_correct
            ]
            top1_n_err, top5_n_err = [
                (x / decoupled_preds[1].size(0)) * 100.0 for x in num_topks_n_correct
            ]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
                top1_v_err, top5_v_err = du.all_reduce([top1_v_err.cuda(), top5_v_err.cuda()])
                top1_n_err, top5_n_err = du.all_reduce([top1_n_err.cuda(), top5_n_err.cuda()])
            # Copy the stats from GPU to CPU (sync point).
            top1_v_err, top5_v_err = (top1_v_err.item(), top5_v_err.item())
            top1_n_err, top5_n_err = (top1_n_err.item(), top5_n_err.item())
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_vmeter.iter_toc()
            val_nmeter.iter_toc()
            val_meter.iter_toc()
            # Update and log stats.
            val_vmeter.update_stats(top1_v_err, top5_v_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            val_nmeter.update_stats(top1_n_err, top5_n_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            val_meter.update_stats(top1_err, top5_err, inputs[0].size(0) * max( cfg.NUM_GPUS, 1),)  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_v_acc": top1_v_err, "Val/Top5_v_acc": top5_v_err},
                    global_step=length * cur_epoch + cur_iter,
                )
                writer.add_scalars(
                    {"Val/Top1_n_acc": top1_n_err, "Val/Top5_n_acc": top5_n_err},
                    global_step=length * cur_epoch + cur_iter,
                )
                writer.add_scalars(
                    {"Val/Top1_action_acc": top1_err, "Val/Top5_action_acc": top5_err},
                    global_step=length * cur_epoch + cur_iter,
                )

        val_nmeter.update_predictions(preds, labels)
        val_vmeter.update_predictions(preds, labels)
        val_meter.update_predictions(preds, labels)
        val_vmeter.log_iter_stats(cur_epoch, cur_iter, "verb")
        val_nmeter.log_iter_stats(cur_epoch, cur_iter, "noun")
        val_meter.log_iter_stats(cur_epoch, cur_iter, "action")
        val_meter.iter_tic()
        val_vmeter.iter_tic()
        val_nmeter.iter_tic()
        # inputs, labels = val_loader.next()
        

    # Log epoch stats.
    val_vmeter.log_epoch_stats(cur_epoch, "verb")
    val_nmeter.log_epoch_stats(cur_epoch, "noun")
    val_meter.log_epoch_stats(cur_epoch, "action")
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            if isinstance(cfg.MODEL.NUM_CLASSES, (list, tuple)):
                all_vpreds = [pred.clone().detach() for pred in val_vmeter.all_vpreds]
                all_vlabels = [label.clone().detach() for label in val_vmeter.all_vlabels]
                all_npreds = [pred.clone().detach() for pred in val_nmeter.all_npreds]
                all_nlabels = [label.clone().detach() for label in val_nmeter.all_nlabels]
                all_preds = [pred.clone().detach() for pred in val_meter.all_npreds]
                all_labels = [label.clone().detach() for label in val_meter.all_nlabels]
                
                if cfg.NUM_GPUS:
                    all_vpreds = [pred.cpu() for pred in all_vpreds]
                    all_vlabels = [label.cpu() for label in all_vlabels]
                    all_npreds = [pred.cpu() for pred in all_npreds]
                    all_nlabels = [label.cpu() for label in all_nlabels]
                writer.plot_eval(
                    preds=all_vpreds, labels=all_vlabels, global_step=cur_epoch
                )
                writer.plot_eval(
                    preds=all_npreds, labels=all_nlabels, global_step=cur_epoch
                )
            else:
                all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
                all_labels = [
                    label.clone().detach() for label in val_meter.all_labels
                ]
                if cfg.NUM_GPUS:
                    all_preds = [pred.cpu() for pred in all_preds]
                    all_labels = [label.cpu() for label in all_labels]
                writer.plot_eval(
                    preds=all_preds, labels=all_labels, global_step=cur_epoch
                )
    val_meter.reset()
    val_vmeter.reset()
    val_nmeter.reset()


def calculate_and_update_precise_bn(cfg, loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if isinstance(inputs, list) and (cfg.MODEL.MODEL_NAME == "TSM" or cfg.MODEL.MODEL_NAME == 'MSG3D'):
                inputs = inputs[0]
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs
    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # for name, param in model.named_parameters():
    #     # prompt and head should be simultaneously fine-tuned
    #     if 'prompt' in name or 'head' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None)
    # print(start_epoch)
    if start_epoch >= 10:
        start_epoch = 0
    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    if cfg.TEST.ENABLE:
        test_loader = loader.construct_loader(cfg, "test")

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)
    if cfg.TEST.ENABLE:
        test_meter = ValMeter(len(test_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    # import pdb; pdb.set_trace()
    epoch_timer = EpochTimer()
    global prompt_enable
    prompt_enable = bool(cfg.PROMPT.ENABLE)
    # eval_epoch(test_loader, model, test_meter, start_epoch, cfg, len(test_loader), writer)
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cur_epoch > 40:
            cfg.TRAIN.EVAL_PERIOD = 1
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg, len(train_loader), writer
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                cfg,
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, scaler if cfg.TRAIN.MIXED_PRECISION else None,)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, len(val_loader), writer)
            if cfg.TEST.ENABLE:
                # pretest_loader = loader.DataPrefetcher(test_loader)
                eval_epoch_without(test_loader, model, test_meter, cur_epoch, cfg, len(test_loader), writer)
                eval_epoch(test_loader, model, test_meter, cur_epoch, cfg, len(test_loader), writer)

    if writer is not None:
        writer.close()
