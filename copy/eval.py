import argparse
import os
import cv2
import numpy as np
import torch
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor,tensor2img
import torchvision.transforms as transforms
from basicsr.metrics import calculate_ssim, calculate_psnr
import time
import logging

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='real_vsr', help='The name of this experiment')
    parser.add_argument('--dataset', type=str, default='REDS4', help='The name of adopted model')
    parser.add_argument('--save_place', type=str, default='results', help='save place')

    opts = parser.parse_args()

    gt_root = '/home/jq/Real/MGLD-VSR-main/dataset/SPMCS/GT'
    sr_root = '/home/jq/Real/MGLD-VSR-main/results/SPMCS'

    opts.save_dir = opts.save_place + '/' + opts.name
    os.makedirs(opts.save_dir,exist_ok=True)
    log_file = os.path.join(opts.save_dir,
                            f"test_{opts.dataset}_{get_time_str()}.log")

    opts.logger = get_root_logger(
        logger_name='Video_Process', log_level=logging.INFO, log_file=log_file)


    video_list = sorted(os.listdir(sr_root))
    calculate_metric = True
    PSNR = 0.0
    SSIM = 0.0

    log_str = f"{opts}\n"
    for video in video_list:
        print(video)
        log_str += f"metrics on {video}\t"
        sr_video_path = os.path.join(sr_root, video)
        gt_video_path = os.path.join(gt_root, video)
        frame_list = sorted(os.listdir(sr_video_path))
        all_len = len(frame_list)
        img_srs = []
        img_gts = []

        for tmp_id, frame in enumerate(frame_list):
            img_gt_path = os.path.join(gt_root, video, frame)
            img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            img_sr_path = os.path.join(sr_root, video, frame)
            img_sr = cv2.imread(img_sr_path)
            # img_sr = img_sr.astype(np.float32) / 255.
            img_srs.append(img_sr)

        PSNR_this_video = [calculate_psnr(sr, gt) for sr, gt in zip(img_srs, img_gts)]
        SSIM_this_video = [calculate_ssim(sr, gt) for sr, gt in zip(img_srs, img_gts)]

        video_psnr = sum(PSNR_this_video) / len(PSNR_this_video)
        video_ssim = sum(SSIM_this_video) / len(SSIM_this_video)

        log_str += f'\t # PSNR: {video_psnr:.4f}\t'
        log_str += f'\t # SSIM: {video_ssim:.4f}\n'
        print(log_str)

        PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
        SSIM += sum(SSIM_this_video) / len(SSIM_this_video)

    PSNR /= len(video_list)
    SSIM /= len(video_list)

    log_str += f"Validation on {sr_root}\n"
    log_str += f'\t # PSNR: {PSNR:.4f}\n'
    log_str += f'\t # SSIM: {SSIM:.4f}\n'
    opts.logger.info(log_str)
    print(log_str)