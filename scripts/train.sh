#!/usr/bin/env bash
DISPLAY_PORT=8097

G='phd'
D='conv'
loadSize=256

#hyper-parameters
L_S=1.2    
L_C=10    
L_GAN=10
L_mask=1
L_tv=1e-4
lr=2e-4
batchs=16
load_iter=0

# network design
use_dist=0
is_matting=1
is_skip=1
is_fft=1
fft_num=2
patch_num=4

datasetmode=new
content_dir="" # trainset 
style_dir="" # trainset
NAME="" # the folder where the model will be saved
checkpoint=''./checkpoint''

CMD="python ../main.py \
--name $NAME \
--checkpoints_dir $checkpoint \
--netG $G \
--netD $D \
--dataset_mode $datasetmode \
--content_dir $content_dir \
--style_dir $style_dir \
--is_train 1 \
--use_dist $use_dist \
--gan_mode wgangp \
--normD batch \
--normG batch \
--preprocess none \
--niter 100 \
--niter_decay 100 \
--input_nc 3 \
--batch_size $batchs \
--num_threads 8 \
--print_freq 400 \
--display_freq 100 \
--save_latest_freq 1000 \
--gpu_ids 0 \
--lambda_g $L_GAN \
--lambda_style $L_S \
--lambda_content $L_C \
--lambda_mask $L_mask  \
--lambda_tv $L_tv \
--is_matting $is_matting \
--is_skip $is_skip \
--is_fft $is_fft \
--fft_num $fft_num \
--patch_number $patch_num \
--lr $lr \
--load_size $loadSize \
--load_iter $load_iter  \
"
echo $CMD
eval $CMD
