# **Human Pose Estimation**

In this work we aim to track body motions in real time using novel dress (called MOCAP suit) and deep learning models. **[Report](./Report.pdf)**

## Checkerboard Pattern

The MOCAP suit is made from the checkerboard pattern. Script to generate can be found [here](./Checkerboard-pattern/arucopp.py).

<p align="center" height = "50%">
    <img width="30%" height = "250px" src="./samples/patterns/sample_pattern1.png">     
    <img width="30%" height = "250px" src="./samples/patterns/sample_pattern2.png">
    <img width="30%" height = "250px" src="./samples/patterns/sample_pattern3.png">
</p>
---

## Checkerboard Dataset

The dataset to train DL models is created with the help of Coco API. Script to generate can be found [here](./Checkerboard-Dataset/).

**Images**

<p align="center" height = "50%">
    <img width="128px" height = "auto" src="./samples/dataset/sample_img1.jpg">     
    <img width="128px" height = "auto" src="./samples/dataset/sample_img2.jpg">
    <img width="128px" height = "auto" src="./samples/dataset/sample_img3.jpg">
    <img width="128px" height = "auto" src="./samples/dataset/sample_img4.png">     
    <img width="128px" height = "auto" src="./samples/dataset/sample_img5.png">
    <img width="128px" height = "auto" src="./samples/dataset/sample_img6.png">
</p>



**Labels**

<p align="center" height = "50%">
    <img width="128px" height = "auto" src="./samples/dataset/sample_lbl1.jpg">     
    <img width="128px" height = "auto" src="./samples/dataset/sample_lbl2.jpg">
    <img width="128x" height = "auto" src="./samples/dataset/sample_lbl3.jpg">
    <img width="128px" height = "auto" src="./samples/dataset/sample_lbl4.png">     
    <img width="128px" height = "auto" src="./samples/dataset/sample_lbl5.png">
    <img width="128px" height = "auto" src="./samples/dataset/sample_lbl6.png">
</p>


---
## Checkerboard Segmentation

UNet performance is shown below.

**Images**

<p align="center" height = "50%">
    <img width = "128px" src="./samples/Predictions/Prediction_1_img.png">
    <img width = "128px" src="./samples/Predictions/Prediction_2_img.png">
    <img width = "128px" src="./samples/Predictions/Prediction_3_img.png">
    <img width = "128px" src="./samples/Predictions/Prediction_4_img.png">
    <img width = "128px" src="./samples/Predictions/Prediction_5_img.png">
    <img width = "128px" src="./samples/Predictions/Prediction_6_img.png">

</p>

**Labels**

<p align="center" height = "50%">
    <img width = "128px" src="./samples/Predictions/Prediction_1_lbl.png">
    <img width = "128px" src="./samples/Predictions/Prediction_2_lbl.png">
    <img width = "128px" src="./samples/Predictions/Prediction_3_lbl.png">
    <img width = "128px" src="./samples/Predictions/Prediction_4_lbl.png">
    <img width = "128px" src="./samples/Predictions/Prediction_5_lbl.png">
    <img width = "128px" src="./samples/Predictions/Prediction_6_lbl.png">

</p>

**Predictions**

<p align="center" height = "50%">
    <img width = "128px" src="./samples/Predictions/Prediction_1_pred.png">
    <img width = "128px" src="./samples/Predictions/Prediction_2_pred.png">
    <img width = "128px" src="./samples/Predictions/Prediction_3_pred.png">
    <img width = "128px" src="./samples/Predictions/Prediction_4_pred.png">
    <img width = "128px" src="./samples/Predictions/Prediction_5_pred.png">
    <img width = "128px" src="./samples/Predictions/Prediction_6_pred.png">

Segmentation models like UNet can be trained using the created checkerboard data from [here](./Checkerboard-segmentation).

## Requirements

- Python 3.6.10
- PyTorch version 1.6.0
- CUDA version 10.1
- 2 NVIDIA® Tesla® V100(16 GB Memory) GPUs.

## References

1. ) Unet Paper: https://arxiv.org/pdf/1505.04597v1.pdf

2. ) Attention UNet Paper: https://arxiv.org/pdf/1804.03999v3.pdf

3. ) Attention Unet Implementation: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets

4. ) ResUnet++ Paper: https://arxiv.org/pdf/1911.07067v1.pdf

5. ) ResUnet++ Implementation: https://github.com/rishikksh20/ResUnet