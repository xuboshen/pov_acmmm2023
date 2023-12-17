#!/usr/bin/env python3

import numpy as np
import csv
import random
import torch
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.datasets import loader
from PIL import Image
from slowfast.models import build_model
import torchvision

def topks_correct(preds, labels, ks):
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    top_max_k_inds = top_max_k_inds.t()
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct

def calTopNAccuracy(preds, labels, ks=(1, )):
    results = []
    rep_max_k_labels = labels.view(-1, 1).expand_as(preds)
    top_max_1_correct = preds.eq(rep_max_k_labels)
    top_max_5_correct = torch.sum(top_max_1_correct, dim=1) > 0.5
    results = [top_max_1_correct[:, :1].float().sum(), top_max_5_correct.float().sum()]
    return results

def decoupleActions(outputs, labels, dataset='assembly101'):
    _, indices = outputs.topk(5, dim=1, largest=True, sorted=True)
    if 'assembly101' in dataset:
        mapping = [torch.zeros(142), torch.zeros(142)]
        action_mapping_filepath = './assembly101_actions.csv'
        with open(action_mapping_filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[1][int(row['action_id'])] = int(row['noun_id'])
                mapping[0][int(row['action_id'])] = int(row['verb_id'])
    elif 'h2o' in dataset:
        mapping = [torch.zeros(36), torch.zeros(36)]
        file = open('./actions.txt', 'r')
        lines = [line.strip('\n').split(' ') for line in file.readlines()]
        for line in lines:
            mapping[0][int(line[0])-1] = int(line[1])-1
            mapping[1][int(line[0])-1] = int(line[2])-1
    verb_preds = mapping[0][indices]
    noun_preds = mapping[1][indices]
    preds = [verb_preds, noun_preds]
    labels = [mapping[0][labels.cpu()], mapping[1][labels.cpu()]]
    return preds, labels

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)

class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return tensor
    
args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)

np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model = build_model(cfg)
model.eval()
if 'h2o' in cfg.TRAIN.DATASET.lower():
    checkpoint = torch.load('./checkpoints/h2o_checkpoint_epoch_00100.pyth')
else:
    checkpoint = torch.load('./checkpoints/ass_checkpoint_epoch_00025.pyth')
model.load_state_dict(checkpoint['model_state_dict'])

input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
normalize = GroupNormalize(input_mean, input_std)
if 'h2o' in cfg.TRAIN.DATASET.lower():
    idx = ['h_480.png', 'h_490.png', 'h_500.png', 'h_510.png']
    label = 10 # place lotion
else:
    idx = ['a_0085.jpg', 'a_0092.jpg', 'a_0099.jpg', 'a_0106.jpg']
    label = 10 # clap hand
images = [Image.open('./imgs/' +   i).convert('RGB') for i in idx]
transform = torchvision.transforms.Compose([
        GroupScale(int(256)),
        GroupCenterCrop(224),
        Stack(roll=(False)),
        ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
        normalize,
    ])
inputs = [transform(images).reshape((3, 4) + (224, 224))]
if isinstance(inputs, (list,)):
    for i in range(len(inputs)):
        inputs[i] = inputs[i].cuda(non_blocking=True)
else:
    inputs = inputs.cuda(non_blocking=True)
# sample test
preds = model(inputs)
preds = preds[0]
print(preds.topk(1, dim=1), label)
