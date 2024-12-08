# Everything Can Be Concealed: Camouflaged Object<br><div align="center">Generation via Finegrained Painting</div>
> Zhenyu Wu, Qiuwei Li, Xuehao Wang, Fengmao Lv
## Get Start

### 1. Prerequisites

- dominate==2.6.0
- numpy==1.19.5
- opencv-python==4.10.0.84 
- Pillow==9.3.0
- scikit_image==0.17.2
- torch==2.0
- pytorch-lightning== 2.3.0 
- visdom==0.1.8.9

### 2. Dataset

Download the following dataset for train:

- [COCO](https://cocodataset.org/#home)

Download the following dataset for test:

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)

### 3. Training

```sh
cd scipts
bash train.sh
```

- `vgg_normalised.pt` is used as the encoder 
- After training, the results models will be saved in `checkpoint` folder

### 4. Testing

```
cd scipts
bash test.sh
```




