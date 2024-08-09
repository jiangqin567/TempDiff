# *TempDiff: Enhancing Temporal-awareness in Latent Diffusion for Real-world Video Super-resolution*
Visual Results Comparison: TecoGAN vs Ours
VideoLQ dataset Sequence 007:

https://github.com/user-attachments/assets/6c6d892c-e693-4e23-8ed0-d0fe73dc71ce

VideoLQ dataset Sequence 008:

https://github.com/user-attachments/assets/75e42059-9e81-48c3-a714-e3fed7ab5b2e

VideoLQ dataset Sequence 033:

https://github.com/user-attachments/assets/224c2eb5-4581-4913-b096-97a54f844d6b

Visual comparison for Color Shift: SD vs Ours
![image](https://github.com/submition/TempDiff/blob/main/color_shift.png)


### Testing
Download the pretrained diffusion denoising U-net and video variational autoencoder from [[BaiduNetDisk]()]. Download the VideoLQ dataset following the links [here](https://github.com/ckkelvinchan/RealBasicVSR). Please update the ckpt_path, load_path and dataroot_gt paths in config files. 

Test on arbitrary size with chopping for VAE.
```
python scripts/vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
  --config configs/unet/tempdiff_unet.yaml \
  --ckpt CKPT_PATH \
  --vqgan_ckpt VQGANCKPT_PATH \
  --seqs-path INPUT_PATH \
  --outdir OUT_DIR \
  --ddpm_steps 50 \
  --dec_w 1.0 \
  --colorfix_type adain \
  --select_idx 0 \
  --n_gpus 1
  
  

### Acknowledgement
This implementation largely depends on [StableSR](https://github.com/IceClear/StableSR). We thank the authors for the contribution.
