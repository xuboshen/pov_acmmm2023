#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import h5py
import torch
import torchvision
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from .transform import *
import slowfast.utils.logging as logging
from . import utils as utils

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
import torch
from torch.utils import data
from tqdm import tqdm
from PIL import Image
import cv2

def read_frames_decord_start_end(video_path, start, end, num_frames, idxs):    
    video_reader = decord.VideoReader(video_path+".mp4", num_threads=8)
    vlen = len(video_reader)
    frame_idxs = idxs
    video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.asnumpy()
    frames = [Image.fromarray(frames[i]).convert('RGB') for i in range(len(idxs))]

    return frames, frame_idxs

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frames(self):
    	return int(self._data[1])

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])
    
    @property
    def verb_label(self):
        return int(self._data[4])
   
    @property
    def noun_label(self):
        return int(self._data[5])
    
    @property
    def id(self):
        return int(self._data[6])
        

@DATASET_REGISTRY.register()
class Assembly101(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        path_annotations = cfg.DATA.PATH_TO_ANNOTATIONS
        self.root_path = cfg.DATA.PATH_TO_DATA_DIR
        self.read_video = False
        self.mode = mode
        self.input_size = 224
        self.cfg = cfg
        self.use_3d_pose = cfg.TRAIN.USE_POSE
        crop_size = self.input_size
        scale_size = self.input_size * 256 // 224
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]
        if cfg.TSM.MODALITY != 'RGBDiff':
            normalize = GroupNormalize(input_mean, input_std)
        else:
            normalize = IdentityTransform()
        if mode == 'train':
            self.test_mode = False
            self.random_shift = True
            train_augmentation = torchvision.transforms.Compose([
                            GroupScale(int(scale_size)),
                            GroupCenterCrop(crop_size),
                            GroupRandomHorizontalFlip(),
                            ])
            self.transform = torchvision.transforms.Compose([
                                    train_augmentation,
                                    Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                                    ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                                    normalize,
                                ])
            if self.read_video:
                if cfg.ASSEMBLY.MODALITY == 'mono':
                    self.list_file = f'{path_annotations}/train_mono.txt'
                if cfg.ASSEMBLY.MODALITY == 'rgb':
                    self.list_file = f'{path_annotations}/train_rgb.txt'
                if cfg.ASSEMBLY.MODALITY == 'combined':
                    self.list_file = f'{path_annotations}/train_combined.txt'
            else:
                self.list_file = f'{path_annotations}/train.txt'
        elif mode == 'val':
            if cfg.DATA.CROPPED:
                train_augmentation = torchvision.transforms.Compose([
                                GroupCenterCrop(crop_size),
                                Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                                normalize,
                                ])
            else:
                self.transform = torchvision.transforms.Compose([
                                GroupScale(int(scale_size)),
                                GroupCenterCrop(crop_size),
                                Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                                normalize,
                            ])
            self.random_shift = False
            self.test_mode = True
            if self.read_video:
                if cfg.ASSEMBLY.MODALITY == 'mono':
                    self.list_file = f'{path_annotations}/validation_mono.txt'
                if cfg.ASSEMBLY.MODALITY == 'rgb':
                    self.list_file = f'{path_annotations}/validation_rgb.txt'
                if cfg.ASSEMBLY.MODALITY == 'combined':
                    self.list_file = f'{path_annotations}/validation_combined.txt'
            else:
                self.list_file = f'{path_annotations}/validation.txt'
        elif mode == 'test':
            self.test_mode = True
            self.random_shift = False
            self.list_file = f'{path_annotations}/test.txt'
            self.transform = torchvision.transforms.Compose([
                    GroupScale(int(scale_size)),
                    GroupCenterCrop(crop_size),
                    Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                    ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                    normalize,
                ])
        elif mode == 'target_domain':
            self.test_mode = True
            self.random_shift = False
            self.list_file = f'{path_annotations}/target_domain.txt'
            self.transform = torchvision.transforms.Compose([
                    GroupScale(int(scale_size)),
                    GroupCenterCrop(crop_size),
                    Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                    ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                    normalize,
                ])
        else:
            raise NotImplementedError("Haven't implement testing yet")
        self.num_segments = cfg.TRAIN.NUM_SEGMENTS
        self.new_length = 1 # 1
        self.modality = cfg.TSM.MODALITY # 

        self._parse_list()

    def _load_image(self, directory, start, end, idx):
        if self.cfg.TRAIN.MULTI_VIEW_FUSION and self.mode == 'train':
            cams = ['C10095_rgb/', 'C10115_rgb/', 'C10118_rgb/', 'C10119_rgb/']
            directory_train = '/'.join(directory.split('/')[:-1])
            img_list = []
            for cam in cams:
                for i in idx:
                    img_list.append(Image.open(os.path.join(directory_train, cam, cam[:-1] + '_' + '%07d.jpg' %i)).convert('RGB'))
            return img_list
        # only for ego2ego upperbound
        elif self.mode == 'train' and self.cfg.PROMPT.ENABLE:
            cams = ['C10095_rgb', 'C10115_rgb', 'C10118_rgb', 'C10390_rgb']
            images = [Image.open(os.path.join(directory, directory.split('/')[-1] + '_' + '%07d.jpg' %i)).convert('RGB') for i in idx]
            domain_label = cams.index(directory.split('/')[-1])
            return images, domain_label
        else:
            images = [Image.open(os.path.join(directory, directory.split('/')[-1] + '_' + '%07d.jpg' %i)).convert('RGB') for i in idx]
            return images# , domain_label

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        for x in tmp:
            x[0] = self.root_path + x[0]
        self.video_list = [VideoRecord(x) for x in tmp]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        
        return offsets + record.start_frames

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + record.start_frames

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        process_data, targets = self.get(record, segment_indices)
        if self.cfg.DATA_LOADER.RETURN_IDX:
            return process_data, targets, index
        else:   
            return process_data, targets

    def get(self, record, indices):
        process_data = None
        if self.mode == 'train':
            if not self.cfg.PROMPT.ENABLE:
                images = self._load_image(record.path, record.start_frames, record.start_frames + record.num_frames, indices)
            else:
                images, domain_label = self._load_image(record.path, record.start_frames, record.start_frames + record.num_frames, indices)
        else:
            images = self._load_image(record.path, record.start_frames, record.start_frames + record.num_frames, indices)
        if self.cfg.TRAIN.MULTI_VIEW_FUSION and self.mode == 'train':
            process_data = self.transform(images).reshape((3, self.num_segments * 4) + (self.input_size, self.input_size))
        else:
            process_data = self.transform(images).reshape((3, self.num_segments) + (self.input_size, self.input_size))
            
        process_data = utils.pack_pathway_output(self.cfg, process_data)
        if isinstance(self.cfg.MODEL.NUM_CLASSES, (tuple, list)):
            if self.mode == 'train':
                if self.cfg.PROMPT.ENABLE:
                    targets = [record.verb_label, record.noun_label, domain_label]
                else:
                    targets = [record.verb_label, record.noun_label]
            else:
                targets = [record.verb_label, record.noun_label]
        else:
            if self.mode == 'train':
                if self.cfg.PROMPT.ENABLE:
                    targets = [record.label, domain_label]
                else:
                    targets = record.label
            else:
                targets = record.label
        return process_data, targets
    
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.video_list)

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.video_list)
