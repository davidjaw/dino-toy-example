# MNIST Classification with DINOv2

## Overview

This project adapts the self-supervised DINOv2 method to a Convolutional Neural Network (CNN) for digit classification on the MNIST dataset. It demonstrates the training process both with and without DINOv2 pre-training. The results indicate that incorporating DINOv2 pre-training enhances accuracy and speeds up convergence in the classification task. In this implementation, we exclusively utilize the DINOb2-Loss.

![](https://i.imgur.com/gZCmv4w.png)

## Environment and Acknowledgements
The project was developed and tested in the following environment:
- Python 3.10
- PyTorch 2.0
The codebase significantly leverages the official implementation of DINOv2, with additional support from ChatGPT and GitHub Copilot.

For more details on DINOv2, refer to the [official repository](https://github.com/facebookresearch/dinov2).



