#!/usr/bin/env bash
DISPLAY_PORT=8097


G='phd'
loadSize=256


# network design
is_matting=1
is_skip=1
is_fft=1
fft_num=2
patch_size=4

batchs=1
test_epoch=50

#####network design
datasetmode=new
content_dir="" # testset
style_dir="" # testset
NAME="last.ckpt" 
checkpoint=''./chectpoint'' # the folder where the model is located

CMD="python ../main.py \
--name $NAME \
--checkpoints_dir $checkpoint \
--netG $G \
--dataset_mode $datasetmode \
--content_dir $content_dir \
--style_dir $style_dir \
--is_train 0 \
--display_id 0 \
--gan_mode wgangp \
--normD batch \
--normG batch \
--preprocess none \
--input_nc 3 \
--batch_size $batchs \
--num_threads 6 \
--print_freq 400 \
--display_freq 1 \
--gpu_ids 0,1 \
--load_size $loadSize \
--is_matting $is_matting \
--is_skip $is_skip \
--is_fft $is_fft \
--epoch $test_epoch  \
--patch_number $patch_size \
--fft_num $fft_num \
"
echo $CMD
eval $CMD
