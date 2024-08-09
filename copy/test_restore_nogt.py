import sys
import os
import cv2
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
import argparse
import yaml
import torchvision.transforms as transforms
from basicsr.data.film_dataset import resize_240_short_side

sys.path.append(os.path.dirname(sys.path[0]))

import torch

from basicsr.data.dataset import Film_dataset
from basicsr.data.film_dataset import Film_test_dataset
from basicsr.utils.util import worker_set_seed, get_root_logger, set_device, frame_to_video,get_time_str
from basicsr.utils.data_util import tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
import logging
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from basicsr.data.util import rgb2lab,rgb2xyz,lab2rgb,lab2xyz,xyz2lab
from torchvision.utils import make_grid

def tensor2img_v1(tensor, out_type=np.float32):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, LAB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, LAB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, LAB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np).round()
    #         Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def Load_model(opts, config_dict):
    net = importlib.import_module('basicsr.models.' + opts.model_name)
    netG = net.InpaintGenerator()
    # netG = net.Video_Backbone()
    # netG = net.BasicVSR()

    # model_path = os.path.join('./pretrained_models', opts.model_name, 'models',
    #                           'net_G_{}.pth'.format(str(opts.which_iter).zfill(5)))
    model_path = os.path.join('./pretrained_models', opts.model_name, 'models',
                              '{}.pth'.format(str(opts.which_iter)))
    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()
    print("Finish loading model ...")
    return netG


def Load_dataset(opts, config_dict):
    val_dataset = Film_test_dataset(config_dict['datasets']['val'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, sampler=None)
    print("Finish loading dataset ...")
    print("Test set statistics:")
    print(f'\n\tNumber of test videos: {len(val_dataset)}')
    return val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='real_old_film_3', help='The name of this experiment')
    parser.add_argument('--model_name', type=str, default='VRT', help='The name of adopted model')
    parser.add_argument('--which_iter', type=str, default='net_G_09000', help='Load which iteraiton')
    # parser.add_argument('--input_video_url', type=str, default=None, help='degraded video input')
    # parser.add_argument('--gt_video_url', type=str, default=None, help='gt video')
    parser.add_argument('--temporal_length', type=int, default=15,
                        help='How many frames should be processed in one forward')
    parser.add_argument('--temporal_stride', type=int, default=10, help='Stride value while sliding window')
    parser.add_argument('--save_image', type=str, default='True', help='save')
    parser.add_argument('--save_place', type=str, default='visual_restore_results', help='save place')

    opts = parser.parse_args()

    # gt_root = '/home/jq/Color/Old_film_restoration/davis/degradation_gt'
    # lq_root = '/home/jq/Color/Old_film_restoration/davis/degradation_lq'
    # gt_root = '/home/jq/Color/Old_film_restoration/REDS4/degradation_gt'
    # lq_root = '/home/jq/Color/Old_film_restoration/REDS4/degradation_lq'

    root = '/home/jq/Color/Old_film_restoration/'
    gt_root = root + opts.name + '/' + 'degradation_gt_full'
    lq_root = root + opts.name + '/' + 'degradation_lq_full'

    with open(os.path.join('./configs', opts.model_name + '.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    val_frame_num = opts.temporal_length
    temporal_stride = opts.temporal_stride
    opts.save_dir = opts.save_place + '/' + opts.model_name
    os.makedirs(opts.save_dir,exist_ok=True)
    log_file = os.path.join(opts.save_dir,
                            f"test_{opts.name}_{get_time_str()}.log")

    opts.logger = get_root_logger(
        logger_name='Video_Process', log_level=logging.INFO, log_file=log_file)

    model = Load_model(opts, config_dict)
    video_list = sorted(os.listdir(lq_root))
    calculate_metric = True
    PSNR = 0.0
    SSIM = 0.0

    log_str = f"{opts}\n"
    for video in video_list:
        print(video)
        log_str += f"Validation on {video}\t"
        lq_video_path = os.path.join(lq_root, video)
        gt_video_path = os.path.join(gt_root, video)
        frame_list = sorted(os.listdir(lq_video_path))
        all_len = len(frame_list)
        img_lqs = []
        img_gts = []

        for tmp_id, frame in enumerate(frame_list):
            img_gt_path = os.path.join(gt_root, video, frame)
            img_gt = cv2.imread(img_gt_path)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            img_lq_path = os.path.join(lq_root, video, frame)
            img_lq = cv2.imread(img_lq_path)
            img_lq = img_lq.astype(np.float32) / 255.
            img_lqs.append(img_lq)


        img_lqs.extend(img_gts)

        from basicsr.utils.data_util import img2tensor
        img_results = img2tensor(img_lqs)

        transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        for i in range(len(img_results)):
            img_results[i] = transform_normalize(img_results[i])

        img_lqs = torch.stack(img_results[:all_len], dim=0)
        img_gts = torch.stack(img_results[all_len:], dim=0)

        all_output = []
        model.eval()

        for i in range(0, all_len, opts.temporal_stride):
            current_part = {}
            current_part['lq'] = img_lqs[i:min(i + val_frame_num, all_len), :, :, :]
            current_part['gt'] = img_gts[i:min(i + val_frame_num, all_len), :, :, :]
            current_part['key'] = frame
            current_part['frame_list'] = frame_list[i:min(i + val_frame_num, all_len)]
            part_lq = current_part['lq'].cuda()
            part_gt = current_part['gt'].cuda()

            part_lq = part_lq.unsqueeze(0)
            part_gt = part_gt.unsqueeze(0)

            h, w = part_gt.shape[3], part_gt.shape[4]

            with torch.no_grad():

                mod_size_h = config_dict['datasets']['val']['crop_size'][0]
                mod_size_w = config_dict['datasets']['val']['crop_size'][1]

                ####################################################################
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w

                part_lq = torch.cat(
                    [part_lq, torch.flip(part_lq, [3])],
                    3)[:, :, :, :h + h_pad, :]
                part_lq = torch.cat(
                    [part_lq, torch.flip(part_lq, [4])],
                    4)[:, :, :, :, :w + w_pad]

                # part_output, _ = model(part_lq)
                part_output = model(part_lq)
                part_output = part_output[:, :, :, :h, :w]

            # all_output.append(part_output.detach().cpu().squeeze(0))
            if i == 0:
                all_output.append(part_output.detach().cpu().squeeze(0))
            else:
                restored_temporal_length = min(i + val_frame_num, all_len) - i - (val_frame_num - opts.temporal_stride)
                all_output.append(part_output[:, 0 - restored_temporal_length:, :, :, :].detach().cpu().squeeze(0))

            del part_lq
            if (i + val_frame_num) >= all_len:
                break

        val_output = torch.cat(all_output, dim=0).squeeze(0)
        gt = img_gts.squeeze(0)

        val_output = (val_output + 1) / 2
        gt = (gt + 1) / 2

        gt_imgs = []
        sr_imgs = []
        for j in range(len(val_output)):
            gt_imgs.append(tensor2img(gt[j]))
            sr_imgs.append(tensor2img(val_output[j]))


        ## Save the image
        for id, sr_img in zip(frame_list, sr_imgs):
            save_place = os.path.join(opts.save_dir, opts.name, video, id)  # e.g. test_results_20_7000
            dir_name = os.path.abspath(os.path.dirname(save_place))
            os.makedirs(dir_name, exist_ok=True)
            cv2.imwrite(save_place, sr_img)

        if calculate_metric:
            PSNR_this_video = [calculate_psnr(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]
            SSIM_this_video = [calculate_ssim(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]

            video_psnr = sum(PSNR_this_video) / len(PSNR_this_video)
            video_ssim = sum(SSIM_this_video) / len(SSIM_this_video)

            log_str += f'\t # PSNR: {video_psnr:.4f}\t'
            log_str += f'\t # SSIM: {video_ssim:.4f}\n'

            PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
            SSIM += sum(SSIM_this_video) / len(SSIM_this_video)

    if calculate_metric:
        PSNR /= len(video_list)
        SSIM /= len(video_list)

        log_str += f"Validation on {lq_root}\n"
        log_str += f'\t # PSNR: {PSNR:.4f}\n'
        log_str += f'\t # SSIM: {SSIM:.4f}\n'

        opts.logger.info(log_str)
        print(log_str)