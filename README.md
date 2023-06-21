# Locating X-ray coronary angiogram keyframes via long short-term spatiotemporal attention with image-to-patch contrastive learning

By Ruipeng Zhang, Binjie Qin, Jun Zhao, Yueqi Zhu, Yisong Lv, Song Ding.

This repository is the official implementation of ["Locating X-ray coronary angiogram keyframes via long short-term spatiotemporal attention with image-to-patch contrastive learning"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10154029). Our paper is accepted for publication in IEEE Transactions on Medical Imaging.


## Introduction
Locating the start, apex and end keyframes of moving contrast agents for keyframe counting in X-ray coronary angiography (XCA) is very important for the diagnosis and treatment of cardiovascular diseases. To locate these keyframes from the class-imbalanced and boundary-agnostic foreground vessel actions that overlap complex backgrounds, we propose long short-term spatiotemporal attention by integrating a convolutional long short-term memory (CLSTM) network into a multiscale Transformer to learn the segment- and sequence-level dependencies in the consecutive-frame-based deep features. Image-to-patch contrastive learning is further embedded between the CLSTM-based long-term spatiotemporal attention and Transformer-based short-term attention modules. The imagewise contrastive module reuses the long-term attention to contrast image-level foreground/background of XCA sequence, while patchwise contrastive projection selects the random patches of backgrounds as convolution kernels to project foreground/background frames into different latent spaces. A new XCA video dataset is collected to evaluate the proposed method. The experimental results show that the proposed method achieves a mAP (mean average precision) of 72.45% and a F-score of 0.8296, considerably outperforming the state-of-the-art methods.

<div align="center">
  <img src="introduction.png" width="600px"/>
</div>

## Code Overview
The structure of this repository is based on [Actionformer](https://github.com/happyharrycn/actionformer_release), one of the first Transformer-based model for temporal action localization. Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the code.

## Data Preparation
**Download Features and Annotations**
* Contact Prof. Qin (bjqin@sjtu.edu.cn) to obtain authorization to download and use our data file *vessel*.
* The file includes vessel features extracted by [SVS-Net](https://pdf.sciencedirectassets.com/271125/1-s2.0-S0893608020X00062/1-s2.0-S0893608020301672/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBgaCXVzLWVhc3QtMSJIMEYCIQCuHPwbyATZKjG8AKDdowvQvJVacvanAJYFoYTp%2B1V%2BuQIhAK5MQk7DzC9IMg3%2FfvhBENbiSuA%2BBW9s83EXBTAV3OL4KrIFCHEQBRoMMDU5MDAzNTQ2ODY1IgzD051C%2BTO1mdcSiqEqjwUeN5Prrithf1kEHD4FpqKZoANq6DkQvKDNu4plrZUqj5%2BW71Ik10XvZ9yqaFgBeF9ceXAh4%2BrigXC%2FZfCEF1ZSGmOlWvHupm7X%2F75NpXqMenQ8%2FQ9gv1q71qGOVrLqLHUpqbYInYi%2Fkecau1kI5GOQGruW5sFm%2Bff1BxkwmqS%2BcBl2DuFLlAGZfXsZgy7CRky7bRiCFCtRSVsi813USHc9aRdRE12Lz%2FflazSVzxBDDQDeUcbGeWomatu01%2FNPNzstp5QQ%2Bqw5J1gpOGgNIdiTEqb1SVlUvtqad9p5lwYvgEowRW%2FLDQrmwlWRjX2eKQxouM9GpZEwrqFoI1t7L763p4yPCE4jSvPjpYC4SaUOmSLKqaeOvqJEldLK2Bz%2FnsN79UrpUu%2B9kYFNK3q%2BzfwXrNcXx%2BkjTGa9p0Iynl2A6JQb%2Buq2JWIGAeLsp2sPmVuIxx3p%2FjZ0OeTJxe2ytRWCqQwLXLKVhfa%2Fk70wZn2RlrGfDfSTdGbMjlhyaC0aLCTT2ZkROpSuWibNLGIYJtL97W7I3psUInRnAPOZtEIqdycDJaduA7D0Bo0yiYVOu7LWv7sYJSSkY1kQ2Xf0E7XbsVx7GbHThuOSQGyaL4K8x%2BjJIFW1gQkKLBc3GJwRlHcmcTw%2Bl9o5WrI98%2FXzFPpcQdYtg%2BnApDLlsp4yNivBNG9pRej%2FImj4tZUU3C%2FN1bMMdR9iJlLyR1%2FRXebaUM%2BM6lZEA3BNx4sCXahqBmDb%2Fym8yphpYmLSWTJpv8%2BuJpUQakFFuj4K7myVUyFR7CDJs7hXVhRbK%2FG90oGbQx%2Bf4sksMLqg9X3pzGrqXa7AzWKcjo7MMfNH1UxVuvy2KaRpuoAosduxr7YG6RpuoSMYMMzluqQGOrABmfdhsJaxQAqoohfM069LoMnw3FTfjVFy7cGVR2rLeDvyKRZf90oMbDjB4pBiKRkqeuuuP9YlQWIPjGzSTThemSmGfms9v51qRvumL9pvdhlqk0%2B1yeFHlf75B7XbG%2FyoTWUaLLVsHPsP3fqjcYv8FXLh4zKvei8A%2B%2BCQGeEJzP%2BIRd275q%2BOO12npkS777oh7mxIYOL9E%2Bwp8Hwt5e3Le6AHVxU84nCMucRTaelQVkU%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230618T081810Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYPFNSYUD%2F20230618%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=dce1d554d6c4ea8984b729b60080426c150f324deadbb317189d5986fd1ce55f&hash=2cd1d9e100688a81aeea4ae45011c0d6070a92f465ef7b18002e212e894f7703&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0893608020301672&tid=spdf-ca82db57-b4fd-4f2b-9c12-bec366e45e74&sid=b6c6c4883afb074498897de41163762420a3gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=050d59035f515e5f57&rr=7d92199eefbd107e&cc=cn) and annotations in json format (similar to ActivityNet annotation format).

**Unpack Features and Annotations**
* Unpack the file under *./data*.
* The folder structure should look like
```
This folder
└───README.md
│   ...
│
└───data
│    └───vessel
│    │	 └───annotations
│    │	     └───vessel.json
│    │	 └───image
│    │	     └───trainx.npy
│    │	     └───valx.npy
│    │	     └───testx.npy
│    └───...
|
└───libs
│
│   ...
```

## Training and Evaluation
**Download Trained Model**

We provide a trained model for this research. The model with all training logs can be downloaded from [model](https://pan.baidu.com/s/1v-ILF-ZukWXuk-JdO02V7Q), extraction coda: `sjtu`. You can also directly train the model yourself without downloading and unpacking.

**Unpack Trained Model**
* Unpack the file under *./ckpt*.
* The folder structure should look like
```
This folder
└───README.md
│   ...
│
└───ckpt
│    └───vessel_SpTeAttenRPCon_reproduce
│    │	 └───logs
│    │	 └───config.txt
│    │	 └───epoch_xxx.pth.tar
│    └───...
|
└───libs
│
│   ...
```

**Training**

Train our model with vessel features. This will create an experiment folder under *./ckpt* that stores training config, logs, and checkpoints.
```shell
python ./train.py ./configs/vessel_SpTeAttenRPCon.yaml --output reproduce
```

**Validation and Testing**

Select the trained model using validation set. The result will show the index of the selected model and the metrics of the selected model.
```shell
python val_and_test_multiprocess.py
```

**Evaluation**

Evaluate the trained model. The expected average mAP should be around 72(%) as in Table 1 of our main paper.
```shell
python ./eval.py ./configs/vessel_SpTeAttenRPCon.yaml ./ckpt/vessel_SpTeAttenRPCon_reproduce/epoch_049.pth.tar
```
* The results (mAP at tIoUs) should be

| Method            |  0.3  |  0.4  |  0.5  |  0.6  |  0.7  |  Avg  |
|-------------------|-------|-------|-------|-------|-------|-------|
| AFSD              | 73.87 | 56.75 | 35.40 | 14.53 |  2.93 | 36.70 |
| TALLFormer        | 70.75 | 69.47 | 57.81 | 38.04 | 17.56 | 50.73 |
| E2E-TAD           | 83.27 | 74.13 | 57.94 | 42.51 | 19.22 | 55.41 |
| Actionformer      | 90.85 | 85.25 | 70.45 | 52.56 | 32.62 | 66.35 |
| Ours              | **98.44** | **92.93** | **80.91** | **53.85** | **36.10** | **72.45** |

* Precision (P), recall (R), F-score (F), average deviation (AD) and confidence interval (CI) will also be reported.

| Method            |  P  |  R  |  F  |  AD  |  CI  |
|-------------------|-------|-------|-------|-------|-------|
| Actionformer      | 0.8096 | 0.8356 | 0.8013 | 5.46 | 0-6.13 |
| Ours              | **0.8342** | **0.8612** | **0.8296** | **4.71** | **0-5.30** |


## Contact
Binjie Qin (bjqin@sjtu.edu.cn)

Ruipeng Zhang (juipengchang@sjtu.edu.cn)

## References

The authors thank all cited authors for providing the source code used in this work, especially [Actionformer](https://github.com/happyharrycn/actionformer_release) and [ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch/tree/master).

If you are using our code, please consider citing our paper.
```
@article{Zhang2023LocatingXC,
  title={Locating X-ray coronary angiogram keyframes via long short-term spatiotemporal attention with image-to-patch contrastive learning.},
  author={Ruipeng Zhang and Binjie Qin and Jun Zhao and Yueqi Zhu and Yisong Lv and Song Ding},
  journal={IEEE transactions on medical imaging},
  year={2023},
  volume={PP}
}
```

If you are using the structure of this repository, you can also cite
```
@inproceedings{zhang2022actionformer,
  title={ActionFormer: Localizing Moments of Actions with Transformers},
  author={Zhang, Chen-Lin and Wu, Jianxin and Li, Yin},
  booktitle={European Conference on Computer Vision},
  series={LNCS},
  volume={13664},
  pages={492-510},
  year={2022}
}
```

If you are using vessel features extracted by SVS-Net, please cite
```
@article{Hao2020SequentialVS,
  title={Sequential vessel segmentation via deep channel attention network},
  author={Dongdong Hao and Song Ding and Linwei Qiu and Yisong Lv and Baowei Fei and Yueqi Zhu and Binjie Qin},
  journal={Neural networks : the official journal of the International Neural Network Society},
  year={2020},
  volume={128},
  pages={172-187}
}
```
