<div align="center">
<h1>Everything Can Be Concealed: Camouflaged Object Generation via Finegrained Painting</h1>
</div>

![arch](assets/1.png)

## Get Start

### 1. Prerequisites

- dominate==2.6.0
- numpy==1.19.5
- opencv-python==4.10.0.84 
- pillow==9.3.0
- scikit_image==0.17.2
- torch==2.0
- pytorch-lightning== 2.3.0 
- visdom==0.1.8.9

### 2. Dataset

Download the following dataset for training:

- [COCO](https://cocodataset.org/#home)

Download the following dataset for testing:

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)

### 3. Training

```
cd scipts
bash train.sh
```

- Pretrained `vgg_normalised.pt` is needed as the encoder 
- After training, the results models will be saved in `checkpoint` folder

### 4. Testing

```
cd scipts
bash test.sh
```




