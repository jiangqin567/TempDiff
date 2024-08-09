# Copyright (c) OpenMMLab. All rights reserved.
import io
import logging
import random

import cv2
import numpy as np

from basicsr.utils import random_mixed_kernels

try:
    import av
    has_av = True
except ImportError:
    has_av = False


if __name__ == '__main__':
    import torch
    from basicsr.utils.options import yaml_load

    yaml_path = '/home/yangxi/projects/StableSR/basicsr/data/data_opts/realbasicvsr_config.yaml'
    opt = yaml_load(yaml_path)

    h, w, c = 512, 512, 3
    inp_path = '/home/yangxi/projects/StableSR/inputs/test_example/OST_120.png'
    img_b = cv2.imread(inp_path)[:, :, [2, 1, 0]]
    inp_dict = {'img': [img_b]}

    # degradations
    random_blur = RandomBlur(
        params=opt['degradation_2']['random_blur']['params'],
        keys=opt['degradation_2']['random_blur']['keys']
    )
    random_resize = RandomResize(
        params=opt['degradation_2']['random_resize']['params'],
        keys=opt['degradation_2']['random_resize']['keys']
    )
    random_noise = RandomNoise(
        params=opt['degradation_2']['random_noise']['params'],
        keys=opt['degradation_2']['random_noise']['keys']
    )
    random_jpeg = RandomJPEGCompression(
        params=opt['degradation_2']['random_jpeg']['params'],
        keys=opt['degradation_2']['random_jpeg']['keys']
    )
    random_mpeg = RandomVideoCompression(
        params=opt['degradation_2']['random_mpeg']['params'],
        keys=opt['degradation_2']['random_mpeg']['keys']
    )
    resize_final = RandomResize(
        params=opt['degradation_2']['resize_final']['params'],
        keys=opt['degradation_2']['resize_final']['keys']
    )
    blur_final = RandomBlur(
        params=opt['degradation_2']['blur_final']['params'],
        keys=opt['degradation_2']['blur_final']['keys']
    )
    out_dict = random_blur(inp_dict)
    out_dict = random_resize(out_dict)
    out_dict = random_noise(out_dict)
    out_dict = random_jpeg(out_dict)
    out_dict = random_mpeg(out_dict)
    out_dict = blur_final(resize_final(out_dict))

    img_a = out_dict['img'][0]
    out_path = '/home/yangxi/projects/StableSR/inputs/test_example/OST_120_Noisy.png'
    cv2.imwrite(out_path, img_a[:, :, [2, 1, 0]])

