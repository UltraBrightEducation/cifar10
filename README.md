# CIFAR10 ConvNet

## About
This python library designed for use in the Trudient Inc. AI Experience Module 4 - Computer Vision lab.

The lab exercises can be found in `notebooks/CIFAR10.ipynb`

The objectives is to train and evaluate convolutional neural networks for classifying images in the CIFAR10 dataset.


## Installation
Before executing the jupyter notebook, please set up a virtual environment and run
`pip install -r requirements.txt` to install all dependencies.


## Experiments

|   model|        hyperparameters | val_acc | val_loss |
| ------ | ---------------------- | ------- | -------- |
|convnet4| simple_convnet_v2.json| 0.7836| 0.6548| 
|convnet6 | simple_convnet_v1.json|0.7738| 0.7539|
|convnet6 | simple_convnet_v3.json|0.8857| 0.4685|