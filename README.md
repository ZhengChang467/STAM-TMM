# STAM (IEEE Transactions on Multimedia 2022)

Zheng Chang,
Xinfeng Zhang,
Shanshe Wang,
Siwei Ma,
Wen Gao.

Official PyTorch Code for **"STAM: A SpatioTemporal Attention based Memory for Video Prediction"** [[paper]](https://ieeexplore.ieee.org/abstract/document/9695337/)

### Requirements
- PyTorch 1.7
- CUDA 11.0
- CuDNN 8.0.5
- python 3.6.7

### Installation
Create conda environment:
```bash
    $ conda create -n STAM python=3.6.7
    $ conda activate STAM
    $ pip install -r requirements.txt
    $ conda install pytorch==1.7 torchvision cudatoolkit=11.0 -c pytorch
```
Download repository:
```bash
    $ git clone git@github.com:ZhengChang467/STAM_TMM.git
```
Unzip MovingMNIST Dataset:
```bash
    $ cd data
    $ unzip mnist_dataset.zip
```
### Test
set --is_training to False in configs/mnist.py and run the following command:
```bash
    $ python STAM_run.py
```
### Train
set --is_training to True in configs/mnist.py and run the following command:
```bash
    $ python STAM_run.py
```
We plan to share the train codes for other datasets soon!
### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
@article{chang2022stam,
  title={STAM: A SpatioTemporal Attention based Memory for Video Prediction},
  author={Chang, Zheng and Zhang, Xinfeng and Wang, Shanshe and Ma, Siwei and Gao, Wen},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```
### License
See [MIT License](https://github.com/ZhengChang467/STAM-TMM/blob/master/LICENSE)

