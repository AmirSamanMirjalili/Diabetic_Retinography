# Diabetic Retinopathy Detection 🔬
This project is an implementation of the paper ["Hybrid machine learning architecture for automated detection and grading of retinal images for diabetic retinopathy"](https://doi.org/10.1117/1.JMI.7.3.034501) using MATLAB for the detection and grading of Diabetic Retinopathy (DR) in retinal fundus images. Diabetic Retinopathy is a leading cause of vision loss and blindness in people with diabetes, and early detection can significantly reduce the risk of vision loss.
## Features
- Automated detection and grading of Diabetic Retinopathy using MATLAB.
- Integration of multiple deep learning models (VGG16 and InceptionV3) for feature extraction.
- Ensemble deep learning approach for improved classification accuracy.
- Utilizes the APTOS 2019 Diabetic Retinopathy dataset.
## System Overview
- Image Preprocessing: Input retinal fundus images are preprocessed to improve image quality using MATLAB's image processing toolbox.
- Feature Extraction: Pre-trained VGG16 and InceptionV3 models extract deep features from preprocessed images. Features from both models are combined to form a composite feature vector.
- Classification: The composite feature vector is fed into a Random Forest classifier for predicting the DR severity stage. (haven't done yet)
## Requirements
To run this project, you'll need:
- MATLAB R2020a or higher
- MATLAB's Deep Learning Toolbox
- MATLAB's Image Processing Toolbox
- Download these CNN's within matlab: Resnet18,Vgg16,Inception-v3,AlexNet
## Usage
- Download and unzip the [APTOS 2019 Diabetic Retinopathy](https://www.kaggle.com/datasets/andrewmvd/aptos2019) dataset.
- Run the DR_Detection.mlx script to start the learning workflow on the APTOS dataset.

## Results
- Read the Report.pdf for the final results
- Class Activation Mapping Results are depicted in the following figure
 ![image](https://github.com/AmirSamanMirjalili/Diabetic_Retinography/assets/57065409/e14f258e-71ac-4cf8-9534-47402c1493c5)


## Acknowledgments
Special thanks to Barath Narayanan Narayanan et al. for their research and the maintainers of the APTOS 2019 Diabetic Retinopathy dataset.

## Future work
A python version of this project is in the future work list
