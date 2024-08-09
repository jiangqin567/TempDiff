 python main.py \
  --train \
  --base configs/video_autoencoder/video_autoencoder_kl_64x64x4_resi.yaml \
  --gpus 0, \
  --name real_vsr \
  --scale_lr False