"""MViT model"""

import math
from functools import partial
from operator import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import numpy as np
import slowfast.utils.logging as logging
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.attention_mprompt import MultiScaleBlock
from slowfast.models.common import TwoStreamFusion
from slowfast.models.utils import Config
from slowfast.models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
)
from functools import reduce
from . import head_helper, operators, resnet_helper, stem_helper
from .build import MODEL_REGISTRY
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None
from towhee.models.layers.layers_with_relprop import GELU, Linear, Dropout, Conv2d, Conv3d, Softmax, Sigmoid, LayerNorm, \
    Einsum, Add, Clone, MaxPool3d, IndexSelect
@MODEL_REGISTRY.register()
class MPromptViT_stage2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_rev = cfg.MVIT.REV.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        self.W = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        self.camera_model_enable = cfg.TDTRL.Enable
        self.tdtrl_layers = cfg.TDTRL.LAYERS
        # used for inference, should merge all
        if self.cfg.TRAIN.MULTI_VIEW_FUSION:
            pass
        self.verify_ideas = False

        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        # patch_embedding
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches
        # we could ignore
        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims
        self.num_prompt_tokens = self.cfg.PROMPT.NUM_TOKENS
        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                separate_qkv=cfg.MVIT.SEPARATE_QKV,
                num_prompt_tokens=self.num_prompt_tokens[i]*2+1 if self.cfg.PROMPT.AGNOSTIC and self.cfg.PROMPT.SPECIFIC else self.num_prompt_tokens[i]+1
            )

            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)
            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]

            embed_dim = dim_out
        
        self.norm = norm_layer(embed_dim)
        self.index_select = IndexSelect()
        self.head = head_helper.TransformerBasicHead(
            2 * embed_dim
            if ("concat" in cfg.MVIT.REV.RESPATH_FUSE and self.enable_rev)
            else embed_dim,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            cfg=cfg,
        )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        if isinstance(cfg.MODEL.NUM_CLASSES, (tuple, list)):
            self.head.projection_v.weight.data.mul_(head_init_scale)
            self.head.projection_v.bias.data.mul_(head_init_scale)
            self.head.projection_n.weight.data.mul_(head_init_scale)
            self.head.projection_n.bias.data.mul_(head_init_scale)

        # self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)
        # get prompts
        self.prompt_dropout = nn.Dropout(self.cfg.PROMPT.DROPOUT)
        self.out_thw = None
        if self.cfg.PROMPT.PROJECT_DIM is not None:
            prompt_dim = self.cfg.PROMPT.PROJECT_DIM
            # initial EMBED_DIM, we could have 4 prompts
            self.prompt_proj = nn.ModuleList([nn.Linear(prompt_dim, prompt_dim),
                                            nn.Linear(prompt_dim*2, prompt_dim*2),
                                            nn.Linear(prompt_dim*4, prompt_dim*4),
                                            nn.Linear(prompt_dim*8, prompt_dim*8)])
            nn.init.kaiming_normal(self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = self.cfg.MVIT.EMBED_DIM
            self.prompt_proj = nn.Identity()
        # initiate prompt
        self.specific_prompt_enable = self.cfg.PROMPT.SPECIFIC
        self.agnostic_prompt_enable = self.cfg.PROMPT.AGNOSTIC
        if self.cfg.PROMPT.INITIALIZATION == 'random':
            # 1, num_of_domains, num_of_tokens, prompt_dim
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_dims, 1) + prompt_dim))
            if self.specific_prompt_enable:
                self.prompt_specific_embeddings = nn.Parameter(torch.zeros(1, 4, self.num_prompt_tokens[0], prompt_dim))
                nn.init.uniform_(self.prompt_specific_embeddings.data, -val, val)
            # xavier_uniform initailzation
            if self.agnostic_prompt_enable:
                self.prompt_agnostic_embeddings = nn.Parameter(torch.zeros(1, self.num_prompt_tokens[0], prompt_dim))
                nn.init.uniform_(self.prompt_agnostic_embeddings.data, -val, val)

            if self.cfg.PROMPT.DEEP:
                dimension = self.cfg.MVIT.EMBED_DIM
                # iter_dims = [(0, dimension)] # (0, 96), (1, 192), (3, 384), (14, 768)
                if self.specific_prompt_enable:
                    self.deep_specific_prompt_embeddings = nn.ParameterList()
                if self.agnostic_prompt_enable:
                    self.deep_agnostic_prompt_embeddings = nn.ParameterList()
                if self.cfg.PROMPT.STAGE2:
                    self.projection_head = nn.ModuleList()
                    # firstly try diverse the in-layer prompts instead of global prompts
                    self.projection_head.append(nn.Linear(dimension * 2, dimension* 2))
                for idx_emb, (layer_i, multiple) in enumerate(self.cfg.MVIT.DIM_MUL):
                    val = math.sqrt(6. / float(3 * reduce(mul, self.patch_dims, 1) + dimension))
                    if self.specific_prompt_enable:
                        self.deep_specific_prompt_embeddings.append(nn.Parameter(torch.zeros(1, 4, self.num_prompt_tokens[layer_i], int(dimension))))
                        nn.init.uniform_(self.deep_specific_prompt_embeddings[idx_emb].data, -val, val)
                    if self.agnostic_prompt_enable:
                        self.deep_agnostic_prompt_embeddings.append(nn.Parameter(torch.zeros(1, self.num_prompt_tokens[layer_i], int(dimension))))
                        nn.init.uniform_(self.deep_agnostic_prompt_embeddings[idx_emb].data, -val, val)
                    dimension *= multiple
                    # if self.cfg.PROMPT.STAGE2:
                    #     # if idx_emb == 2:
                    #     #     self.projection_head.append(nn.Linear(int(dimension) // 2, int(dimension) // 2))
                    #     # else:
                    #     self.projection_head.append(nn.Linear(int(dimension), int(dimension)))
        else:
            raise ValueError("other initalization scheme is not supported")

    def incorporate_prompt(self, x, domain_labels):
        B = x.shape[0]
        # after CLS token, all before image patches
        x, bcthw = self.patch_embed(x)
        if self.specific_prompt_enable:
            if not self.training:
                # evaluation in stage1
                prompt_specific_emb = self.prompt_dropout(self.prompt_proj(self.prompt_specific_embeddings[0, :, :, :].mean(0).expand(B, -1, -1)))
            else:
                # training
                if not self.cfg.PROMPT.STAGE2:
                    # stage1
                    prompt_specific_emb = self.prompt_dropout(self.prompt_proj(self.prompt_specific_embeddings[0, domain_labels, :, :]))
                else:
                    # stage2
                    prompt_specific_emb = self.prompt_dropout(self.prompt_proj(self.prompt_specific_embeddings[0, :, :, :].mean(0).expand(B, -1, -1)))
        if self.agnostic_prompt_enable:
            prompt_agnostic_emb = self.prompt_dropout(self.prompt_proj(self.prompt_agnostic_embeddings.expand(B, -1, -1)))
        # cls, specific, agnostic
        if self.specific_prompt_enable and not self.agnostic_prompt_enable:
            x = torch.cat((
                prompt_specific_emb,
                x
            ), dim=1)
        elif self.agnostic_prompt_enable and not self.specific_prompt_enable:
            x = torch.cat((
                prompt_agnostic_emb,
                x
            ), dim=1)
        elif self.agnostic_prompt_enable and self.specific_prompt_enable:
            x = torch.cat((
                prompt_specific_emb,
                prompt_agnostic_emb,
                x
            ), dim=1)
        else:
            raise NotImplementedError("Not Implemented Prompt type")
        return x, bcthw

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward_deep_prompt(self, x, thw, domain_labels):
        B = x.shape[0]
        cnt = 0
        out_prompts = []
        for blk_idx, blk in enumerate(self.blocks):
            # # (0, 96), (1, 192), (3, 384), (14, 768)
            if blk_idx in [1, 3, 14]:
                if self.specific_prompt_enable:
                    if not self.training:
                        # evaluation
                        deep_specific_prompt_embeddings = self.prompt_dropout(self.prompt_proj(self.deep_specific_prompt_embeddings[cnt][0, :, :, :].mean(0).expand(B, -1, -1)))
                    else:
                        # training
                        if self.cfg.PROMPT.STAGE2:
                            deep_specific_prompt_embeddings = self.prompt_dropout(self.prompt_proj(self.deep_specific_prompt_embeddings[cnt][0, :, :, :].mean(0).expand(B, -1, -1)))
                        else:
                            deep_specific_prompt_embeddings = self.prompt_dropout(self.prompt_proj(self.deep_specific_prompt_embeddings[cnt][0, domain_labels, :, :]))
                if self.agnostic_prompt_enable:
                    deep_agnostic_prompt_embeddings = self.prompt_dropout(self.prompt_proj(self.deep_agnostic_prompt_embeddings[cnt].expand(B, -1, -1)))
                cnt += 1
                if self.specific_prompt_enable and not self.agnostic_prompt_enable:
                    x = torch.cat((
                        x[:, :1, :],
                        deep_specific_prompt_embeddings,
                        x[:, (1 + self.num_prompt_tokens[blk_idx - 1]):, :]
                    ), dim=1)
                elif self.agnostic_prompt_enable and not self.specific_prompt_enable:
                    x = torch.cat((
                        x[:, :1, :],
                        deep_agnostic_prompt_embeddings,
                        x[:, (1 + self.num_prompt_tokens[blk_idx - 1]):, :]
                    ), dim=1)
                elif self.specific_prompt_enable and self.agnostic_prompt_enable:
                    x = torch.cat((
                        x[:, :1, :],
                        deep_specific_prompt_embeddings,
                        deep_agnostic_prompt_embeddings,
                        x[:, (1 + self.num_prompt_tokens[blk_idx - 1] * 2):, :]
                    ), dim=1)
                else:
                    raise NotImplementedError("Not Implemented Prompt type")
            x, thw = blk(x, thw)
            if self.cfg.PROMPT.STAGE2 and self.cfg.PROMPT.SSDIV:
                if blk_idx in [0, 2, 13, 15]:
                    if self.specific_prompt_enable and not self.agnostic_prompt_enable:
                        out_prompts.append(self.projection_head[cnt](x[:, 1: 1 + self.num_prompt_tokens[blk_idx], :]))
                    elif self.agnostic_prompt_enable and not self.specific_prompt_enable:
                        out_prompts.append(self.projection_head[cnt](x[:, 1: 1 + self.num_prompt_tokens[blk_idx], :]))
                    elif self.specific_prompt_enable and self.agnostic_prompt_enable:
                        out_prompts.append(self.projection_head[cnt](x[:, 1: 1 + self.num_prompt_tokens[blk_idx] * 2, :]))
                    else:
                        raise NotImplementedError("Not Implemented Prompt type")
        self.out_thw = thw
        if self.cfg.PROMPT.SSDIV:
            return x, out_prompts
        else:
            return x

    def forward(self, x, bboxes=None, return_attn=False, return_feat=False, alpha=None, domain_labels=None):
        x = x[0]
        x, bcthw = self.incorporate_prompt(x, domain_labels)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape
        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        # normal encoder process
        if self.cfg.PROMPT.DEEP and self.cfg.PROMPT.SSDIV:
            x, out_prompts = self.forward_deep_prompt(x, thw, domain_labels)
        else:
            x = self.forward_deep_prompt(x, thw, domain_labels)

        if self.use_mean_pooling:
            if self.cls_embed_on:
                x = x[:, 1:]
            x = x.mean(1)
            x = self.norm(x)
        elif self.cls_embed_on:
            x = self.norm(x)
            x = self.index_select(x, dim=1, indices=torch.tensor(0, device=x.device))
            # x = x[:, 0]
            x = x.squeeze(1)
        else:  # this is default, [norm->mean]
            x = self.norm(x)
            x = x.mean(1)

        pred = self.head(x)
        if return_feat is True:
            if self.cfg.PROMPT.STAGE2 and self.cfg.PROMPT.SSDIV:
                return x, pred, out_prompts
            else:
                return x, pred
        else:
            if self.cfg.PROMPT.STAGE2 and self.cfg.PROMPT.SSDIV:
                return pred, out_prompts
            else:
                return pred

    def train(self, mode=True):
        if self.cfg.PROMPT.STAGE2 and not self.cfg.TRAIN.ADA:
            if mode:
                if self.cfg.TRAIN.SHOT:
                    # SHOT
                    self.patch_embed.train()
                    self.blocks.train()
                    self.prompt_proj.train()
                    self.prompt_dropout.train()
                    self.head.eval()
                else:
                    # prompt tuning
                    self.patch_embed.eval()
                    self.blocks.eval()
                    self.prompt_proj.train()
                    self.prompt_dropout.train()
                    self.head.train()
                
            else:
                for module in self.children():
                    module.train(mode)
        else:
            for module in self.children():
                module.train(mode)
        self.training = mode
    
    def relprop(self, cam: torch.Tensor, method: str = "rollout", start_layer: int = 0, **kwargs) -> torch.Tensor:
        cam = self.head.relprop(cam, **kwargs)
        if self.cls_embed_on:
            cam = cam.unsqueeze(1)
            cam = self.index_select.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        thw = self.out_thw
        for blk in reversed(self.blocks):
            cam, thw = blk.relprop(cam, thw=thw, **kwargs)
        
        # cam rollout
        attn_cams = []
        for blk in self.blocks:
            attn_heads = blk.attn.get_attn_cam().clamp(min=0)
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            attn_cams.append(avg_heads)
        cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
        cam = cam[:, 0, 1:]
        return cam

def find_most_h_w(all_layer_matrices):
    shape_info = torch.vstack(
        [torch.as_tensor([layer_matrix.shape[-2] for layer_matrix in all_layer_matrices]),
         torch.as_tensor([layer_matrix.shape[-1] for layer_matrix in all_layer_matrices])])
    return torch.mode(shape_info).values[0], torch.mode(shape_info).values[1]


def resize_last_dim_to_most(layer_matrix, last_dim_size):
    if layer_matrix.shape[-1] == last_dim_size:
        return layer_matrix
    else:
        cls_token, attn = layer_matrix[..., 1], layer_matrix[..., 1:]
        if attn.shape[-1] > last_dim_size - 1:
            assert attn.shape[-1] % (last_dim_size - 1) == 0
        else:
            assert (last_dim_size - 1) % attn.shape[-1] == 0
        factor = (last_dim_size - 1) / attn.shape[-1]
        attn = torch.nn.functional.interpolate(attn, size=last_dim_size - 1, mode="nearest")
        attn = attn * factor
        return torch.cat([cls_token.unsqueeze(dim=-1), attn], dim=-1)

def align_scale(all_layer_matrices):
    most_attn_h, most_attn_w = find_most_h_w(all_layer_matrices)
    aligned_layer_matrices = []
    for layer_matrix in all_layer_matrices:
        layer_matrix = resize_last_dim_to_most(layer_matrix, most_attn_w)
        layer_matrix = layer_matrix.permute(0, 2, 1)
        layer_matrix = resize_last_dim_to_most(layer_matrix, most_attn_h)
        layer_matrix = layer_matrix.permute(0, 2, 1)
        aligned_layer_matrices.append(layer_matrix)
    return aligned_layer_matrices
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    all_layer_matrices = align_scale(all_layer_matrices)
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention
