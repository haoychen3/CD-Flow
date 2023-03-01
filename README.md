# Learning a Deep Color Difference Metric for Photographic Images

## Introduction
This repository contains the official pytorch implementation of the paper ["Learning a Deep Color Difference Metric for Photographic Images"](https://openreview.net) by Zhihua Wang, Haoyu Chen, Yang Yang, Qilin Sun, and Kede Ma, IEEE Conference on Computer Vision and Pattern Recognition, 2023.

Most well-established and widely used color difference (CD) metrics are handcrafted and subject-calibrated against uniformly colored patches, which do not generalize well to photographic images characterized by natural scene complexities. Developing CD formulae for photographic images is still an active research topic in imaging/illumination, vision science, and color science communities. In this paper, we aim to learn a deep CD metric for photographic images with four desirable properties. First, it well aligns with the observations in vision science that color and form are linked inextricably invisual cortical processing. Second, it is a proper metric in the mathematical sense. Third, it computes accurate CDs between photographic images, differing mainly in color appearances.  Fourth, it is robust to mild geometric distortions (e.g.,translation or due to parallax), which are often present in photographic images of the same scene captured by different digital cameras. We show that all these properties can be satisfied simultaneously by learning a multi-scale autoregressive normalizing flow for feature transform, after which the Euclidean distance is linearly proportional to the human perceptual CD.

## Prerequisites
* python 3.10

* pytorch 1.12.0

* ``pip install -r requirements.txt``

## Training
To train the CD-Flow from scratch, execute the following commands:
```bash
python main.py --training_datadir path/to/the/dataset --work_path work_dir --datapath data --batch_size_train 8
```
For the [SPCD dataset](https://arxiv.org/abs/2205.13489), you can download via [Baidu Netdisk](https://pan.baidu.com/s/18bzu-qhpMW3PqLTlVdoZRQ?pwd=txeh) or [Google Drive](https://drive.google.com/drive/folders/1Wh9fcDPviZcYWqCpXvnsJux1mnZ5WkCf?usp=share_link).
## Evaluation
To evaluate the STRESS, PLCC and SRCC of your checkpoints on test set, execute:
```bash
python test.py --training_datadir path/to/the/dataset --work_path work_dir --datapath data --batch_size_train 8
```
## Citation

## Acknowledgment
