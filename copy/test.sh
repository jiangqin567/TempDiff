python vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
  --dataset SPMCS \
  --config configs/mgldvsr/mgldvsr_512_realbasicvsr_deg.yaml \
  --ckpt /home/jq/Real/MGLD-VSR-main/checkpoints/MGLD-VSR-Pretrained-Models/mgldvsr_unet.ckpt \
  --vqgan_ckpt /home/jq/Real/MGLD-VSR-main/checkpoints/MGLD-VSR-Pretrained-Models/video_vae_cfw.ckpt \
  --seqs-path  /home/jq/Real/MGLD-VSR-main/dataset/SPMCS/GT_degrade \
  --outdir results \
  --ddpm_steps 50 \
  --dec_w 1.0 \
  --colorfix_type adain \
  --select_idx 0 \
  --n_gpus 1


#/home/jq/Trans/VSR-Transformer-main/data/REDS4/val/val_sharp_bicubic/X4/ \
#   /home/jq/Real/RealBasicVSR-master/data/VideoLQ/Input
#/home/jq/Trans/VSR-Transformer-main/data/UDM10/BDx4/
#/home/jq/Real/MGLD-VSR-main/dataset/UDM10/GT_degrade_sub