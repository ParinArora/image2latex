#!/bin/bash
#PBS -N image_encoder_gpu_resnet18_2
#PBS -o image_encoder_gpu_resnet18_2.out
#PBS -e image_encoder_gpu_resnet18_2.err

# ask for exactly 1 chunk on compute3, with 8 CPUs, 1 GPU, 32 GB RAM
# and pin it to host compute3
# note: "host=compute3" is a PBS‐Pro extension
# (some sites may omit it, in which case PBS will pick any eligible node)
#PBS -l select=1:ncpus=16:mem=32gb:host=compute4
#PBS -l walltime=168:00:00
#PBS -q gpu

module load compiler/anaconda3
source /apps/compilers/anaconda3/bin/activate
conda activate pyoccenv

cd $PBS_O_WORKDIR
CUDA_LAUNCH_BLOCKING=1 
# python train_latex_vae.py
python train_image_encoder_rms.py   --latex_file /home/parin.arora_ug2023/text2latex/PRINTED_TEX_230k/final_png_formulas.txt   --png_names_file /home/parin.arora_ug2023/text2latex/PRINTED_TEX_230k/corresponding_png_images.txt    --png_dir /home/parin.arora_ug2023/text2latex/PRINTED_TEX_230k/generated_png_images   --vae_ckpt checkpoints/best_latex_vae.pth   --resume_from image_encoder_resnet18_epoch_07.pth   --kl_weight 0.05   --warmup_ratio 0.2