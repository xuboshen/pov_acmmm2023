#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""
import os

import torch
from fvcore.common.registry import Registry

import slowfast.models.optimizer as optim

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""
global prompt_enable
prompt_enable = None
def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    global prompt_enable
    prompt_enable = False if bool(cfg.PROMPT.ENABLE) is False else True
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."
    # gpu_id = os.environ["LOCAL_RANK"]
    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    # import pdb; pdb.set_trace()
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.MODEL.ARCH in ['uniformer']:
        checkpoint = model.get_pretrained_model(cfg)
        if checkpoint:
            # logger.info('load pretrained model')
            model.load_state_dict(checkpoint, strict=False)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
    return model
'''
def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."
    # gpu_id = os.environ["LOCAL_RANK"]
    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    # import pdb; pdb.set_trace()
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    try:
        from apex import amp
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')     
        print("apex used for acceleration, fp16 used")   
    except ImportError as e:
        raise ("error :", e)
        print("apex not found, fp16 not used.")
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
    return model, optimizer
'''

def build_model_for_multimodal(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."
    # gpu_id = os.environ["LOCAL_RANK"]
    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    # import pdb; pdb.set_trace()
    rgb_model = MODEL_REGISTRY.get(name[0])(cfg)
    pose_model = MODEL_REGISTRY.get(name[1])(cfg)
    model = [rgb_model, pose_model]

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        if isinstance(model, list):
            for mo in model:
                mo = mo.cuda(device=cur_device)
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    try:
        from apex import amp
        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')     
        print("apex used for acceleration, fp16 used")   
    except ImportError as e:
        raise ("error :", e)
        print("apex not found, fp16 not used.")
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        # for mo in model:
        model[0] = torch.nn.parallel.DistributedDataParallel(
            module=model[0], device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
        model[1] = torch.nn.parallel.DistributedDataParallel(
            module=model[1], device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
    return model, optimizer
