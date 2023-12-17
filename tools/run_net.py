#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg

from test_net import test
from train_net import train
from validation_net import validation


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    global prompt_enable
    prompt_enable = True if cfg.PROMPT.ENABLE else False
    # Perform vpn training.
    if cfg.DISTILLATION.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=dctrain)
    if cfg.VPN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train_vpnpp)
    # Perform training.
    if cfg.MULTITRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=multitrain)

    # if cfg.DISTILLATION.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=distillationtrain)

    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)
    
    '''
    if cfg.TRAIN.DANN:
        launch_job(cfg=cfg, init_method=args.init_method, func=trainDANN)
    '''
    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        # launch_job(cfg=cfg, init_method=args.init_method, func=test)
        launch_job(cfg=cfg, init_method=args.init_method, func=validation)

    # Perform model visualization.
    # if cfg.TENSORBOARD.ENABLE and (
    #     cfg.TENSORBOARD.MODEL_VIS.ENABLE
    #     or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    # ):
    #     launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # # Run demo.
    # if cfg.DEMO.ENABLE:
    #     demo(cfg)


if __name__ == "__main__":
    global prompt_enable
    main()
