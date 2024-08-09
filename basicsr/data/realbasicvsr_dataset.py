import cv2
import math
import time
import os
import os.path as osp
import numpy as np
import random
import torch

from copy import deepcopy
from pathlib import Path
from torch.utils import data as data

from basicsr.data.mmcv_transforms import Clip, UnsharpMasking, RescaleToZeroOne
from basicsr.data.mmcv_transforms import RandomBlur, RandomResize, RandomNoise, RandomJPEGCompression, RandomVideoCompression, DegradationsWithShuffle
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment, single_random_crop, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img, imwrite
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY


# @DATASET_REGISTRY.register()
class RealVSRRecurrentDataset(data.Dataset):

    def __init__(self, opt):
        super(RealVSRRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_gt'])
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

        # the first degradation
        self.random_blur_1 = RandomBlur(
            params=opt['degradation_1']['random_blur']['params'],
            keys=opt['degradation_1']['random_blur']['keys']
        )
        self.random_resize_1 = RandomResize(
            params=opt['degradation_1']['random_resize']['params'],
            keys=opt['degradation_1']['random_resize']['keys']
        )
        self.random_noise_1 = RandomNoise(
            params=opt['degradation_1']['random_noise']['params'],
            keys=opt['degradation_1']['random_noise']['keys']
        )
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt['degradation_1']['random_jpeg']['params'],
            keys=opt['degradation_1']['random_jpeg']['keys']
        )
        self.random_mpeg_1 = RandomVideoCompression(
            params=opt['degradation_1']['random_mpeg']['params'],
            keys=opt['degradation_1']['random_mpeg']['keys']
        )

        # the second degradation
        self.random_blur_2 = RandomBlur(
            params=opt['degradation_2']['random_blur']['params'],
            keys=opt['degradation_2']['random_blur']['keys']
        )
        self.random_resize_2 = RandomResize(
            params=opt['degradation_2']['random_resize']['params'],
            keys=opt['degradation_2']['random_resize']['keys']
        )
        self.random_noise_2 = RandomNoise(
            params=opt['degradation_2']['random_noise']['params'],
            keys=opt['degradation_2']['random_noise']['keys']
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt['degradation_2']['random_jpeg']['params'],
            keys=opt['degradation_2']['random_jpeg']['keys']
        )
        self.random_mpeg_2 = RandomVideoCompression(
            params=opt['degradation_2']['random_mpeg']['params'],
            keys=opt['degradation_2']['random_mpeg']['keys']
        )

        # final
        self.resize_final = RandomResize(
            params=opt['degradation_2']['resize_final']['params'],
            keys=opt['degradation_2']['resize_final']['keys']
        )
        self.blur_final = RandomBlur(
            params=opt['degradation_2']['blur_final']['params'],
            keys=opt['degradation_2']['blur_final']['keys']
        )

        # transforms
        self.usm = UnsharpMasking(
            kernel_size=opt['transforms']['usm']['kernel_size'],
            sigma=opt['transforms']['usm']['sigma'],
            weight=opt['transforms']['usm']['weight'],
            threshold=opt['transforms']['usm']['threshold'],
            keys=opt['transforms']['usm']['keys']
        )
        self.clip = Clip(keys=opt['transforms']['clip']['keys'])
        self.rescale = RescaleToZeroOne(keys=opt['transforms']['rescale']['keys'])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the GT frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'


            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=False)
            img_gts.append(img_gt)

        # randomly crop
        img_gts = single_random_crop(img_gts, gt_size, img_gt_path)

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])
        img_lqs = deepcopy(img_gts)

        out_dict = {'lqs': img_lqs, 'gts': img_gts}

        out_dict = self.usm.transform(out_dict)

        ## the first degradation
        out_dict = self.random_blur_1(out_dict)
        out_dict = self.random_resize_1(out_dict)
        out_dict = self.random_noise_1(out_dict)
        out_dict = self.random_jpeg_1(out_dict)
        out_dict = self.random_mpeg_1(out_dict)

        ## the second degradation
        out_dict = self.random_blur_2(out_dict)
        out_dict = self.random_resize_2(out_dict)
        out_dict = self.random_noise_2(out_dict)
        out_dict = self.random_jpeg_2(out_dict)
        out_dict = self.random_mpeg_2(out_dict)

        ## final resize
        out_dict = self.resize_final(out_dict)
        out_dict = self.blur_final(out_dict)

        # post process
        out_dict = self.clip(out_dict)
        out_dict = self.rescale.transform(out_dict)

        # list-to-list
        for k in out_dict.keys():
            out_dict[k] = img2tensor(out_dict[k])    #lq: 128*128   gt: 512*512


            #########################saving degradation lq images#############################

            # print(out_dict[k][0].shape,len(out_dict[k]))

        return out_dict

    def __len__(self):
        return len(self.keys)

