# TDCNet
Official repository of the AAAI 2026 paper "Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving Infrared Small Target Detection".

## Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving Infrared Small Target Detection  
Authors: Shukai Guo, Houzhang Fang, Qiuhuan Chen, Yi Chang; Luxin Yan

## Abstract

Moving infrared small target detection (IRSTD) plays a critical role in practical applications, such as surveillance of unmanned aerial vehicles (UAVs) and UAV-based search system. Moving IRSTD still remains highly challenging due to weak target features and complex background interference. Accurate spatio-temporal feature modeling is crucial for moving target detection, typically achieved through either temporal differences or spatio-temporal (3D) convolutions. Temporal difference can explicitly leverage motion cues but exhibits limited capability in extracting spatial features, whereas 3D convolution effectively represents spatio-temporal features yet lacks explicit awareness of motion dynamics along the temporal dimension. In this paper, we propose a novel moving IRSTD network (TDCNet), which effectively extracts and enhances spatio-temporal features for accurate target detection. Specifically, we introduce a novel temporal difference convolution (TDC) re-parameterization module that comprises three parallel TDC blocks designed to capture contextual dependencies across different temporal ranges. Each TDC block fuses temporal difference and 3D convolution into a unified spatio-temporal convolution representation. This re-parameterized module can effectively capture multi-scale motion contextual features while suppressing pseudo-motion clutter in complex backgrounds, significantly improving detection performance. Moreover, we propose a TDC-guided spatio-temporal attention mechanism that performs cross-attention between the spatio-temporal features extracted from the TDC-based backbone and a parallel 3D backbone. This mechanism models their global semantic dependencies to refine the current frame’s features, thereby guiding the model to focus more accurately on critical target regions. To facilitate comprehensive evaluation, we construct a new challenging benchmark, IRSTD-UAV, consisting of 15,106 real infrared images with diverse low signal-to-clutter ratio scenarios and complex backgrounds. Extensive experiments on IRSTD-UAV and public infrared datasets demonstrate that our TDCNet achieves state-of-the-art detection performance in moving target detection. 

## SCINet Framework

![image-20250407200916034](./assets/SCINet_Framework.png)

## Visualization

![image-20250407201214584](./assets/Visualization.png)

## Environment

[PyTorch >= 1.7](https://pytorch.org/)  
[BasicSR >= 1.3.4.9](https://github.com/XPixelGroup/BasicSR)

## Installation

```
pip install -r requirements.txt
python setup.py develop
```

## How To Test

· Refer to ./options/test for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
· The pretrained models are available in ./experiments/pretrained_models/  
· Then run the follwing codes (taking net_g_SCINet_x4.pth as an example):  

```shell
python basicsr/test.py -opt options/test/benchmark_SCINet_x4.yml
```

The testing results will be saved in the ./results folder.

## How To Train

· Refer to ./options/train for the configuration file of the model to train.  
· Preparation of training data can refer to this page. All datasets can be downloaded at the official website.  
· Note that the default training dataset is based on lmdb, refer to [docs in BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) to learn how to generate the training datasets.  
· The training command is like  

```shell
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_SCINet_x4.yml
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/train_SCINet-S_x4.yml --launcher pytorch
```


## Citation
If you find our work useful for your research, please consider citing our paper:
```
@inproceedings{2026AAAI_TDCNet,
  title     = {Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving Infrared Small Target Detection},
  author    = {Houzhang Fang and Shukai Guo and Qiuhuan Chen and Yi Chang and Luxin Yan},
  booktitle   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  pages     = { },
}
```

In additoin to the above paper, please also consider citing the following references. Thank you!
```
@inproceedings{2025CVPR_UniCD,
    title     = {Detection-Friendly Nonuniformity Correction: A Union Framework for Infrared {UAV} Target Detection},
    author    = {Houzhang Fang and Xiaolin Wang and Zengyang Li and Lu Wang and Qingshan Li and Yi Chang and Luxin Yan},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025},
    pages     = {11898-11907},
}
@ARTICLE{2023TII_DAGNet,
  title     =  {Differentiated Attention Guided Network Over Hierarchical and Aggregated Features for Intelligent {UAV} Surveillance},
  author    =  {Houzhang Fang and Zikai Liao and Xuhua Wang and Yi Chang and Luxin Yan},
  journal   =  {IEEE Transactions on Industrial Informatics}, 
  year      =  {2023},
  volume    =  {19},
  number    =  {9},
  pages     =  {9909-9920},
  }
@inproceedings{2023ACMMM_DANet,
title       =  {{DANet}: Multi-scale {UAV} Target Detection with Dynamic Feature Perception and Scale-aware Knowledge Distillation},
author      =  {Houzhang Fang and Zikai Liao and Lu Wang and Qingshan Li and Yi Chang and Luxin Yan and Xuhua Wang},
booktitle   =  {Proceedings of the 31st ACM International Conference on Multimedia (ACMMM)},
pages       =  {2121-2130},
year        =  {2023},
}
@article{2024TGRS_SCINet,
  title     = {{SCINet}: Spatial and Contrast Interactive Super-Resolution Assisted Infrared {UAV} Target Detection},
  author    = {Houzhang Fang and Lan Ding and Xiaolin Wang and Yi Chang and Luxin Yan and Li Liu and Jinrui Fang},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  volume    = {62},
  year      = {2024},
  pages     = {1-22},
}
@ARTICLE{2022TIMFang,
  title     =  {Infrared Small {UAV} Target Detection Based on Depthwise Separable Residual Dense Network and Multiscale Feature Fusion},
  author    =  {Houzhang Fang and Lan Ding and Liming Wang and Yi Chang and Luxin Yan and Jinhui Han},
  journal   =  {IEEE Transactions on Instrumentation and Measurement}, 
  year      =  {2022},
  volume    =  {71},
  number    =  {},
  pages     =  {1-20},
}
```

## Contact
If you have any question, please contact: houzhangfang@xidian.edu.cn,

Copyright &copy; Xidian University.

## Acknowledgments
Some of the SR code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR/). Thanks for their excellent work!

## License
MIT License. This code is only freely available for non-commercial research use.

