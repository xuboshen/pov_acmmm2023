#!/usr/bin/env python3

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode
import torch

def add_custom_config(_C):
    # Assembly101 dataset
    _C.ASSEMBLY = CfgNode()
    _C.ASSEMBLY.MODALITY = 'mono'
    # Assembly101 pose dataset
    _C.ASSEMBLY_pose = CfgNode()
    _C.ASSEMBLY_pose.RANDOM_CHOOSE = False
    _C.ASSEMBLY_pose.RANDOM_SHIFT = False
    _C.ASSEMBLY_pose.RANDOM_MOVE = False
    _C.ASSEMBLY_pose.WINDOW_SIZE = -1
    _C.ASSEMBLY_pose.NORMALIZATION = False
    _C.ASSEMBLY_pose.USE_MMAP = True
    _C.ASSEMBLY.ONLY_POSE = False
    _C.GPU_ID = None
    _C.DEBUG = False
    _C.TRAIN.GRADIENT_ACCUMULATION = 1
    _C.TRAIN.PARTIAL_DS = None
    _C.TRAIN.USE_POSE = False
    _C.TRAIN.MULTITASK = False
    _C.TRAIN.USE_MASK = False
    _C.DATA.PATH_TO_ANNOTATIONS = None
    _C.alpha = 1.
    _C.temperature = 2


    # -----------------------------------------------------------------------------
    # MViT options
    # -----------------------------------------------------------------------------
    _C.MVIT = CfgNode()

    # Options include `conv`, `max`.
    _C.MVIT.MODE = "conv"

    # If True, perform pool before projection in attention.
    _C.MVIT.POOL_FIRST = False

    # If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
    _C.MVIT.CLS_EMBED_ON = True

    # Kernel size for patchtification.
    _C.MVIT.PATCH_KERNEL = [3, 7, 7]

    # Stride size for patchtification.
    _C.MVIT.PATCH_STRIDE = [2, 4, 4]

    # Padding size for patchtification.
    _C.MVIT.PATCH_PADDING = [2, 4, 4]

    # If True, use 2d patch, otherwise use 3d patch.
    _C.MVIT.PATCH_2D = False

    # Base embedding dimension for the transformer.
    _C.MVIT.EMBED_DIM = 96

    # Base num of heads for the transformer.
    _C.MVIT.NUM_HEADS = 1

    # Dimension reduction ratio for the MLP layers.
    _C.MVIT.MLP_RATIO = 4.0

    # If use, use bias term in attention fc layers.
    _C.MVIT.QKV_BIAS = True

    # Drop path rate for the tranfomer.
    _C.MVIT.DROPPATH_RATE = 0.1

    # The initial value of layer scale gamma. Set 0.0 to disable layer scale.
    _C.MVIT.LAYER_SCALE_INIT_VALUE = 0.0

    # Depth of the transformer.
    _C.MVIT.DEPTH = 16

    # Normalization layer for the transformer. Only layernorm is supported now.
    _C.MVIT.NORM = "layernorm"

    # Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
    # the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
    _C.MVIT.DIM_MUL = []

    # Head number multiplication at layer i. If 2.0 is used, then the next block will
    # increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
    _C.MVIT.HEAD_MUL = []

    # Stride size for the Pool KV at layer i.
    # Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
    _C.MVIT.POOL_KV_STRIDE = []

    # Initial stride size for KV at layer 1. The stride size will be further reduced with
    # the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
    _C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

    # Stride size for the Pool Q at layer i.
    # Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
    _C.MVIT.POOL_Q_STRIDE = []

    # If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
    # Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
    _C.MVIT.POOL_KVQ_KERNEL = None

    # If True, perform no decay on positional embedding and cls embedding.
    _C.MVIT.ZERO_DECAY_POS_CLS = True

    # If True, use norm after stem.
    _C.MVIT.NORM_STEM = False

    # If True, perform separate positional embedding.
    _C.MVIT.SEP_POS_EMBED = False

    # Dropout rate for the MViT backbone.
    _C.MVIT.DROPOUT_RATE = 0.0

    # If True, use absolute positional embedding.
    _C.MVIT.USE_ABS_POS = True

    # If True, use relative positional embedding for spatial dimentions
    _C.MVIT.REL_POS_SPATIAL = False

    # If True, use relative positional embedding for temporal dimentions
    _C.MVIT.REL_POS_TEMPORAL = False

    # If True, init rel with zero
    _C.MVIT.REL_POS_ZERO_INIT = False

    # If True, using Residual Pooling connection
    _C.MVIT.RESIDUAL_POOLING = False

    # Dim mul in qkv linear layers of attention block instead of MLP
    _C.MVIT.DIM_MUL_IN_ATT = False

    # If True, using separate linear layers for Q, K, V in attention blocks.
    _C.MVIT.SEPARATE_QKV = False

    # The initialization scale factor for the head parameters.
    _C.MVIT.HEAD_INIT_SCALE = 1.0

    # Whether to use the mean pooling of all patch tokens as the output.
    _C.MVIT.USE_MEAN_POOLING = False

    # If True, use frozen sin cos positional embedding.
    _C.MVIT.USE_FIXED_SINCOS_POS = False
    _C.MVIT._change_model_keys = False


    # -----------------------------------------------------------------------------
    # Masked pretraining options
    # -----------------------------------------------------------------------------
    _C.MASK = CfgNode()

    # Whether to enable Masked style pretraining.
    _C.MASK.ENABLE = False

    # Whether to enable MAE (discard encoder tokens).
    _C.MASK.MAE_ON = False

    # Whether to enable random masking in mae
    _C.MASK.MAE_RND_MASK = False

    # Whether to do random masking per-frame in mae
    _C.MASK.PER_FRAME_MASKING = False

    # only predict loss on temporal strided patches, or predict full time extent
    _C.MASK.TIME_STRIDE_LOSS = True

    # Whether to normalize the pred pixel loss
    _C.MASK.NORM_PRED_PIXEL = True

    # Whether to fix initialization with inverse depth of layer for pretraining.
    _C.MASK.SCALE_INIT_BY_DEPTH = False

    # Base embedding dimension for the decoder transformer.
    _C.MASK.DECODER_EMBED_DIM = 512

    # Base embedding dimension for the decoder transformer.
    _C.MASK.DECODER_SEP_POS_EMBED = False

    # Use a KV kernel in decoder?
    _C.MASK.DEC_KV_KERNEL = []

    # Use a KV stride in decoder?
    _C.MASK.DEC_KV_STRIDE = []

    # The depths of features which are inputs of the prediction head.
    _C.MASK.PRETRAIN_DEPTH = [15]

    # The type of Masked pretraining prediction head.
    # Can be "separate", "separate_xformer".
    _C.MASK.HEAD_TYPE = "separate"

    # The depth of MAE's decoder
    _C.MASK.DECODER_DEPTH = 0

    # The weight of HOG target loss.
    _C.MASK.PRED_HOG = False
    # Reversible Configs
    _C.MVIT.REV = CfgNode()

    # Enable Reversible Model
    _C.MVIT.REV.ENABLE = False

    # Method to fuse the reversible paths
    # see :class: `TwoStreamFusion` for all the options
    _C.MVIT.REV.RESPATH_FUSE = "concat"

    # Layers to buffer activations at
    # (at least Q-pooling layers needed)
    _C.MVIT.REV.BUFFER_LAYERS = []

    # 'conv' or 'max' operator for the respath in Qpooling
    _C.MVIT.REV.RES_PATH = "conv"

    # Method to merge hidden states before Qpoolinglayers
    _C.MVIT.REV.PRE_Q_FUSION = "avg"

    
    # -----------------------------------------------------------------------------
    # 3DTRL options
    # -----------------------------------------------------------------------------
    _C.TDTRL = CfgNode()
    _C.TDTRL.Enable = False
    _C.TDTRL.LAYERS = []
    _C.TDTRL.USE_POSE = False
    _C.TDTRL.LAMBDA = None
    # -----------------------------------------------------------------------------
    # Prompted options
    # -----------------------------------------------------------------------------
    _C.PROMPT = CfgNode()
    _C.PROMPT.ENABLE = False
    _C.PROMPT.LOCATION = 'prepend'
    # tokens used for single view prompt
    _C.PROMPT.NUM_TOKENS = None
    # projection dimension used for projection
    _C.PROMPT.PROJECT_DIM = None
    # used for dropout after projection
    _C.PROMPT.DROPOUT = None
    # Deep prompt 
    _C.PROMPT.DEEP = False
    # initialization strategy
    _C.PROMPT.INITIALIZATION = 'random'
    # initialization strategy
    _C.PROMPT.STAGE2 = False
    _C.PROMPT.AGNOSTIC = False
    _C.PROMPT.SPECIFIC = False
    _C.PROMPT.SSDIV = False
    _C.PROMPT.PL = False
    _C.PROMPT.CONTRASTIVE = False
    # default is block-level
    _C.PROMPT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
    _C.PROMPT.BLOCK_IDX = [1, 3, 14]

    _C.PATCHPROMPT = CfgNode()
    _C.PATCHPROMPT.ENABLE = False
    _C.PATCHPROMPT.PROMPT_SIZE = 20
    _C.PATCHPROMPT.CROPPED_PATH = None

    _C.SHOT = CfgNode()
    _C.SHOT.ENABLE = False
    _C.SHOT.PL = False