#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from .Decoder import ReverseLayerF
from towhee.models.layers.layers_with_relprop import GELU, Linear, Dropout, Conv2d, Conv3d, Softmax, Sigmoid, LayerNorm, \
    Einsum, Add, Clone, MaxPool3d, IndexSelect
class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        ltype='linear',
        cfg=None
    ):
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate)
        if isinstance(num_classes, (list, tuple)):
            self.projection_v = nn.Linear(dim_in, num_classes[0], bias=True)
            self.projection_n = nn.Linear(dim_in, num_classes[1], bias=True)
            # self.projection = nn.Linear(dim_in, num_classes[2], bias=True)
        else:
            if ltype == 'linear':
                self.projection_v = Linear(dim_in, num_classes, bias=True)
            else:
                self.projection_v = torch.nn.utils.weight_norm(nn.Linear(dim_in // 2, num_classes, bias=True), name='weight')
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
        self.teacher = False

    def forward(self, x, alpha=None):
        output = []
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection_v(x)
        if not self.training:
            x = self.act(x)
        output.append(x)
        return output
    