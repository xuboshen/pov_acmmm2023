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
    for cur_iter, (inputs, labels) in enumerate(val_loader):
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

        preds = preds[0]
        if isinstance(labels, list):
            labels = labels[0]
        if isinstance(preds, list):
            preds = preds[0]
        decoupled_preds, decoupled_labels = decoupleActions(preds, labels)
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
        if cfg.DEBUG and cur_iter == 50:
            break
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
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None)
    # print(start_epoch)
    if start_epoch >= 100:
        start_epoch = 0
    # Create the video train and val loaders.
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    # train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)
    # if cfg.TEST.ENABLE:
    #     test_meter = ValMeter(len(test_loader), cfg)

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
    eval_epoch(val_loader, model, val_meter, start_epoch, cfg, len(val_loader), writer)
