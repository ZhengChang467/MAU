# MAU (NeurIPS2021)

Zheng Chang,
Xinfeng Zhang,
Shanshe Wang,
Siwei Ma,
Yan Ye,
Xinguang Xiang,
Wen Gao.

Official PyTorch Code for **"MAU: A Motion-Aware Unit for Video Prediction and
Beyond"** [[paper]](https://proceedings.neurips.cc/paper/2021/file/e25cfa90f04351958216f97e3efdabe9-Paper.pdf)

### Requirements
- PyTorch 1.7
- CUDA 11.0
- CuDNN 8.0.5
- python 3.6.7

### Installation
Create conda environment:
```bash
    $ conda create -n MAU python=3.6.7
    $ conda activate MAU
    $ pip install -r requirements.txt
    $ conda install pytorch==1.7 torchvision cudatoolkit=11.0 -c pytorch
```
Download repository:
```bash
    $ git clone git@github.com:ZhengChang467/MAU.git
```
Unzip MovingMNIST Dataset:
```bash
    $ cd data
    $ unzip mnist_dataset.zip
```
### Test
```bash
    $ python MAU_run.py --is_train False
```
### Train
```bash
    $ python MAU_run.py --is_train True
```
We plan to share the train codes for other datasets soon!
### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
@article{chang2021mau,
title={MAU: A Motion-Aware Unit for Video Prediction and Beyond},
author={Chang, Zheng and Zhang, Xinfeng and Wang, Shanshe and Ma, Siwei and Ye, Yan and Xinguang, Xiang and Gao, Wen},
journal={Advances in Neural Information Processing Systems},
volume={34},
year={2021}}
```
### License
See [MIT License](https://github.com/ZhengChang467/MAU/blob/master/LICENSE)

