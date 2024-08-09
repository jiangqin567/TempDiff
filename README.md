# *TempDiff: Enhancing Temporal-awareness in Latent Diffusion for Real-world Video Super-resolution*
Visual Results Comparison: TecoGAN vs Ours
VideoLQ dataset Sequence 007:

https://github.com/user-attachments/assets/971511a9-5189-42a0-a144-3dda4334f9fe

VideoLQ dataset Sequence 008:

https://github.com/user-attachments/assets/07bf37d0-e84f-46cc-8c72-4ae0bbd1079a

VideoLQ dataset Sequence 033:

https://github.com/user-attachments/assets/afecc981-19b2-4983-b6c9-447502e882ee


### Testing
Download the pretrained diffusion denoising U-net and video variational autoencoder from [[BaiduNetDisk](https://pan.baidu.com/s/1MjdAMO-UE0A5BwMaqalpxQ) code: gji9]. Download the VideoLQ dataset following the links [here](https://github.com/ckkelvinchan/RealBasicVSR). Please update the ckpt_path, load_path and dataroot_gt paths in config files. 

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
