import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import csv


def evaluate(input=None, ref=None, metric_name=None, save_file=None,metric_mode='FR'):
    """Inference demo for pyiqa.
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', type=str, default=None, help='input image/folder path.')
    # parser.add_argument('-r', '--ref', type=str, default=None, help='reference image/folder path if needed.')
    # parser.add_argument(
    #     '--metric_mode',
    #     type=str,
    #     default='FR',
    #     help='metric mode Full Reference or No Reference. options: FR|NR.')
    # parser.add_argument('-m', '--metric_name', type=str, default='PSNR', help='IQA metric name, case sensitive.')
    # parser.add_argument('--save_file', type=str, default=None, help='path to save results.')
    #
    # args = parser.parse_args()

    metric_name = metric_name.lower()

    # set up IQA model
    iqa_model = create_metric(metric_name, metric_mode=metric_mode)
    metric_mode = iqa_model.metric_mode

    if os.path.isfile(input):
        input_paths = [input]
        if ref is not None:
            ref_paths = [ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(input, '*')))
        if ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(ref, '*')))

    # if save_file:
    #     sf = open(save_file, 'w')
    #     sfwriter = csv.writer(sf)

    avg_score = 0
    test_img_num = len(input_paths)
    if metric_name != 'fid':
        pbar = tqdm(total=test_img_num, unit='image')
        for idx, img_path in enumerate(input_paths):
            img_name = os.path.basename(img_path)
            if metric_mode == 'FR':
                ref_img_path = ref_paths[idx]
            else:
                ref_img_path = None

            score = iqa_model(img_path, ref_img_path).cpu().item()
            avg_score += score
            pbar.update(1)
            pbar.set_description(f'{metric_name} of {img_name}: {score}')
            pbar.write(f'{metric_name} of {img_name}: {score}')
            if save_file:
                sfwriter.writerow([img_name, score])
            
        pbar.close()
        avg_score /= test_img_num

    else:
        assert os.path.isdir(input), 'input path must be a folder for FID.'
        avg_score = iqa_model(input, ref)

    
    msg = f'Average {metric_name} score of {input} with {test_img_num} images is: {avg_score}'
    print(msg)
    # if save_file:
    #     sf.close()

    # if save_file:
    #     print(f'Done! Results are in {save_file}.')
    # else:
    #     print(f'Done!')

    return avg_score


if __name__ == '__main__':
    metric_name = 'lpips'
    dir = '/home/jq/Real/MGLD-VSR-main/results/UDM10'
    ref = '/home/jq/Real/MGLD-VSR-main/dataset/UDM10/GT_sub'
    name = dir.split('/')[-1]

    save_file= '/home/jq/Real/MGLD-VSR-main/results/real_vsr' + '/' + name + '_metric.csv'

    if save_file:
        sf = open(save_file, 'w')
        sfwriter = csv.writer(sf)

    video_list = sorted(os.listdir(dir))
    metric_ = []
    for video in video_list:
        video_path = os.path.join(dir, video)
        ref_path = os.path.join(ref, video)
        metric_.append(evaluate(video_path, ref=ref_path, metric_name=metric_name, save_file = save_file))

    metric = sum(metric_) / len(video_list)
    msg = f'Average {metric_name} score of {name} is: {metric}'
    print(msg)

    if save_file:
        sfwriter.writerow([name, metric])

    if save_file:
        sf.close()




# ['ahiq', 'brisque', 'ckdn', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'clipscore', 'cnniqa', 'cw_ssim', 'dbcnn', 'dists', 'entropy', 'fid', 'fsim', 'gmsd', 'hyperiqa', 'ilniqe', 'laion_aes',
# 'liqe', 'liqe_mix', 'lpips', 'lpips-vgg', 'mad', 'maniqa', 'maniqa-kadid', 'maniqa-koniq', 'maniqa-pipal', 'ms_ssim', 'musiq',
# 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-koniq', 'nima-spaq', 'nima-vgg16-ava', 'niqe', 'nlpd', '
# nrqm', 'paq2piq', 'pi', 'pieapp', 'psnr', 'psnry', 'ssim', 'ssimc', 'stlpips', 'stlpips-vgg', 'topiq_fr', 'topiq_fr-pipal', '
# topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 'topiq_nr-face', 'topiq_nr-flive', 'topiq_nr-spaq', 'tres', 'tres-flive', 'tres-koniq',
# 'uranker', 'vif', 'vsi']

#