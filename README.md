# **Human Pose Estimation**

In this work we aim to track body motions in real time using novel dress (called MOCAP suit) and deep learning models. Here we briefly mention about how to create the MOCAP suit, how the training dataset is created, and the results obtained using U-Net. **[Report](./Report.pdf)** includes the exact details of the work.

## [Checkerboard Pattern](./Checkerboard-pattern/arucopp.py)

The MOCAP suit is made from the checkerboard pattern. This a pattern created specifically for real-time detection of human joints (points).

<p align="center" height = "50%">
    <img width="30%" height = "250px" src="./samples/patterns/sample_pattern1.png">     
    <img width="30%" height = "250px" src="./samples/patterns/sample_pattern2.png">
    <img width="30%" height = "250px" src="./samples/patterns/sample_pattern3.png">
</p>


## [Checkerboard Dataset](./Checkerboard-Dataset/)

The dataset to train DL models is created with the help of Coco API. We generate around 30,000 samples with good quality checkerboard pattern placed over various class categories like humans, vehicles, animals, etc.

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


## [Checkerboard Segmentation](./Checkerboard-segmentation)

Different SoTA segmentation models are benchmarked over this dataset. Some predictions of U-Net are shown below. U-Net gives a dice score of **96.7%**.

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


## Requirements

- Python 3.6.10
- PyTorch version 1.6.0
- CUDA version 10.1
- 2 NVIDIA® Tesla® V100(16 GB Memory) GPUs.

