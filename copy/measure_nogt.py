import os
import argparse
import numpy as np

from PIL import Image
# from natsort import natsorted
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import torch
import lpips
import shutil
import sys
import piq
import skvideo.measure

def set_niqe(img):
    if img is None:
        return None
    return skvideo.measure.niqe(img)[0]


def getfilelist(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append(t)
    all_file = sorted(all_file)
    return all_file

def check(a, b):
    clip_name_a = a.split("/")[-2]
    frame_name_a = a.split("/")[-1]
    clip_name_b = b.split("/")[-2]
    frame_name_b = b.split("/")[-1]

    if clip_name_a == clip_name_b and frame_name_a == frame_name_b:
        return True
    print(clip_name_a+"/"+frame_name_a)
    print(clip_name_b+"/"+frame_name_b)
    return False

def img_to_tensor_init(img):
    # image_size_w = img.width
    # image_size_h = img.height
    # image = (np.asarray(img) / 255.0).reshape(image_size_w * image_size_h, 3).transpose().reshape(3, image_size_h, image_size_w)
    # image = (np.asarray(img) / 255.0).reshape(image_size_w * image_size_h, 3).transpose().reshape(3, image_size_h, image_size_w)
    # torch_image = torch.from_numpy(image).float()
    # torch_image = torch_image * 2.0 - 1.0
    # torch_image = torch_image.unsqueeze(0)

    torch_image = torch.tensor(np.asarray(img)).permute(2, 0, 1)[None, ...] / 255.
    torch_image = torch_image * 2.0 - 1.0
    return torch_image


def img_to_tensor(img):
    # image_size_w = img.width
    # image_size_h = img.height
    # image = (np.asarray(img) / 255.0).reshape(image_size_w * image_size_h, 3).transpose().reshape(3, image_size_h, image_size_w)
    # image = (np.asarray(img) / 255.0).reshape(image_size_w * image_size_h, 3).transpose().reshape(3, image_size_h, image_size_w)
    # torch_image = torch.from_numpy(image).float()
    # torch_image = torch_image * 2.0 - 1.0
    # torch_image = torch_image.unsqueeze(0)
    torch_image = torch.tensor(np.asarray(img)).permute(2, 0, 1)[None, ...] / 255.
    torch_image = torch_image * 2.0 - 1.0
    return torch_image

class Reconstruction_Metrics:
    def __init__(self, metric_list=['ssim', 'psnr', 'fid'], data_range=1, win_size=21, multichannel=True):
        self.data_range = data_range
        self.win_size = win_size
        self.multichannel = multichannel
        # self.fid_calculate = FID()
        # self.loss_fn_vgg = lpips.LPIPS(net='alex')
        for metric in metric_list:
            setattr(self, metric, True)

    def calculate_metric(self, fake_image_path):
        """
            inputs: .txt files, floders, image files (string), image files (list)
            gts: .txt files, floders, image files (string), image files (list)
        """

        # fid_value = 0
        brisque = []
        niqe = []

        # image_name_list = [name for name in os.listdir(real_image_path) if
        #                    name.endswith((('.png', '.jpg', '.jpeg', '.JPG', '.bmp')))]


        frame_path_fake = getfilelist(fake_image_path)
        # fake_image_name_list = os.listdir(fake_image_path)
        for i, image_path_fake in enumerate(frame_path_fake):
            path_fake = image_path_fake
            PIL_fake = Image.open(path_fake).convert('RGB')
            fake_torch_image = img_to_tensor(PIL_fake)
            brisque_each_img = np.array(piq.brisque((fake_torch_image + 1)/2, data_range=1., reduction='none')).astype(np.float32)
            niqe_each_img = set_niqe(np.array(PIL_fake.convert('L')).astype(np.float32))

            # print(brisque_each_img)
            # print(niqe_each_img)


            brisque.append(brisque_each_img)
            niqe.append(niqe_each_img)


        print(
              "BRISQUE: %.4f" % np.round(np.mean(brisque), 4),
              "BRISQUE Variance: %.4f" % np.round(np.var(brisque), 4))
        print(
              "NIQE: %.4f" % np.round(np.mean(niqe), 4),
              "NIQE Variance: %.4f" % np.round(np.var(niqe), 4))
        return  np.round(np.mean(brisque), 4), np.round(np.mean(niqe), 4)
        # return  np.round(np.mean(lipis), 4), fid_value, np.round(
        #     np.mean(brisque), 4), np.round(np.mean(niqe), 4)


def get_metric(fake_dir):
    Get_metric = Reconstruction_Metrics()
    brisque_out, niqe_out = Get_metric.calculate_metric(fake_dir)

    save_txt = os.path.join('.', 'metric.txt')
    with open(save_txt, 'a') as txt2:
        txt2.write(" ")
        txt2.write("brisque:")
        txt2.write(str(brisque_out))
        txt2.write(" ")
        txt2.write("niqe:")
        txt2.write(str(niqe_out))
        txt2.write('\n')


if __name__ == "__main__":
    fake_path = '/home/jq/Real/MGLD-VSR-main/results/VideoLQ'
    get_metric(fake_path)


#real_old_film_gt:
# gt: BRISQUE: 57.6770 BRISQUE Variance: 85.5046
# NIQE: 12.1981 NIQE Variance: 0.3419

#input: BRISQUE: 34.4213 BRISQUE Variance: 95.0501
# NIQE: 22.2360 NIQE Variance: 164.6340

#Inpainter_v1: BRISQUE: 26.2313 BRISQUE Variance: 48.7954
# NIQE: 21.9685 NIQE Variance: 179.4910

# BRISQUE: 27.3130 BRISQUE Variance: 49.5699
# NIQE: 22.5190 NIQE Variance: 144.8150
# BRISQUE: 27.2144 BRISQUE Variance: 49.4712
# NIQE: 23.1203 NIQE Variance: 167.6306
# BRISQUE: 27.4696 BRISQUE Variance: 51.5278
# NIQE: 22.0511 NIQE Variance: 124.8778


# RNN_Swin_4: BRISQUE: 19.6207 BRISQUE Variance: 47.2081
# NIQE: 23.0506 NIQE Variance: 198.0846

#Inpainter_v1:
# real_old_film:
# BRISQUE: 40.5997 BRISQUE Variance: 54.6587
# NIQE: 11.7881 NIQE Variance: 0.2369


