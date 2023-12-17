#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import h5py
import random
from itertools import chain as chain
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from numpy.random import randint
import logging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import slowfast.utils.logging as logging
from .transform import *
from . import utils as utils
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class H2o(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, num_retries=10):
        path_annotations = cfg.DATA.PATH_TO_ANNOTATIONS
        self.root_path = cfg.DATA.PATH_TO_DATA_DIR
        self.mode = mode
        self.input_size = 224
        self.cfg = cfg
        self.use_3d_pose = cfg.TRAIN.USE_POSE
        crop_size = self.input_size
        scale_size = self.input_size * 234 // 224
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]
        normalize = GroupNormalize(input_mean, input_std)
        if self.mode == 'train':
            self.random_shift = True
            self.file_list = f"{path_annotations}/train.txt"
            train_augmentation = torchvision.transforms.Compose([
                            GroupScale(int(scale_size)),
                            # GroupCenterCrop(crop_size),
                            GroupMultiScaleCrop(self.input_size, [1, .96, .92]),
                            GroupRandomHorizontalFlip(),
                            GroupColorJitter(),
                            # GroupRandomRotation(60),
                            ])
            self.transform = torchvision.transforms.Compose([
                                train_augmentation,
                                Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                                normalize,
                            ])
            pass
        elif self.mode == 'val':
            self.random_shift = False
            self.file_list = f"{path_annotations}/validation.txt"
            self.transform = torchvision.transforms.Compose([
                        # GroupScale(int(scale_size)),
                        GroupCenterCrop(crop_size),
                        Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                        ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                        normalize,
                    ])
        elif self.mode == 'test':
            self.random_shift = False
            self.file_list = f"{path_annotations}/test.txt"
            self.transform = torchvision.transforms.Compose([
                        # GroupScale(int(scale_size)),
                        GroupCenterCrop(crop_size),
                        Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                        ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                        normalize,
                    ])
        elif self.mode == 'target_domain':
            self.random_shift = False
            self.file_list = f"{path_annotations}/target_domain.txt"
            self.transform = torchvision.transforms.Compose([
                        GroupScale(int(scale_size)),
                        GroupCenterCrop(crop_size),
                        # GroupMultiScaleCrop(self.input_size, [1, .96, .92]),
                        Stack(roll=(cfg.MODEL.ARCH in ['BNInception', 'InceptionV3'])),
                        ToTorchFormatTensor(div=(cfg.MODEL.ARCH not in ['BNInception', 'InceptionV3'])),
                        normalize,
                    ])
        else:
            raise NotImplementedError(f"we don't have test set or {self.mode} set")
        self.num_segments = cfg.TRAIN.NUM_SEGMENTS
        self.new_length = 1
        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.file_list)]
        tmp = tmp[1:] # remove headline
        for x in tmp:
            x[1] = self.root_path + x[1]
        self.video_list = [VideoRecord(x) for x in tmp]

    def __len__(self):
        return len(self.video_list)

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.video_list)

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        # num_frames == 4, average_du = 1
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            # [0, 1, 2, 3] \times avg_d + [0, avg_d) *
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
        if self.mode == 'train':
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        inputs, targets = self.get(record, segment_indices)
        if self.cfg.DATA_LOADER.RETURN_IDX:
            if self.cfg.PATCHPROMPT.ENABLE is True:
                boxes = self.get_boxes(record, index)
                return inputs, targets, boxes, index
            else:
                return inputs, targets, index
        else:
            if self.cfg.PATCHPROMPT.ENABLE is True:
                boxes = self.get_boxes(record, index)
                return inputs, targets, boxes
            else:
                return inputs, targets

    def _load_image(self, directory, start, end, idx):
        # used for prompt
        if self.mode == 'train' and 'upperBound' not in self.cfg.DATA.PATH_TO_ANNOTATIONS:
            cams = ['cam0', 'cam1', 'cam2', 'cam3']
            domain_label = cams.index(directory.split('/')[-1])
            try:
                images = [Image.open(os.path.join(directory, 'rgb/', '%06d.png' %i)).convert('RGB') for i in idx]
            except:
                images = [Image.open(os.path.join(directory, 'rgb/', '%06d.jpg' %i)).convert('RGB') for i in idx]
            return images, domain_label
        else:
            return [Image.open(os.path.join(directory, 'cam4/', 'rgb/', '%06d.png' %i)).convert('RGB') for i in idx]
            
    def get(self, record, indices):
        process_data = None
        if self.mode == 'train':
            images, domain_labels = self._load_image(record.path, record.start_frames, record.start_frames + record.num_frames, indices)
            targets = [record.label, domain_labels]
        else:
            images = self._load_image(record.path, record.start_frames, record.start_frames + record.num_frames, indices)
            targets = record.label
        if self.cfg.TRAIN.MULTI_VIEW_FUSION and self.mode == 'train':
            process_data = self.transform(images).reshape((3, self.num_segments * 4) + (self.input_size, self.input_size))
        else:
            process_data = self.transform(images).reshape((3, self.num_segments) + (self.input_size, self.input_size))
 
        process_data = utils.pack_pathway_output(self.cfg, process_data)
        return process_data, targets

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[1]

    @property
    def start_frames(self):
    	return int(self._data[3])

    @property
    def num_frames(self):
        return int(self._data[4]) - int(self._data[3]) + 1

    @property
    def label(self):
        return int(self._data[2]) - 1
    