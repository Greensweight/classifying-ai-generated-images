# AI-Generated Image Detection

## Abstract

With the rapid advancement in AI-generated images, distinguishing between real and AI-generated images has become a pressing challenge. This project investigates the effectiveness of various machine learning techniques in detecting AI-generated images. We explore three methods—Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Convolutional Neural Networks (CNN)—to determine their effectiveness in image classification tasks.

## Introduction

AI-generated images have reached a level of realism that can make them nearly indistinguishable from authentic images. This raises concerns about image authenticity and trustworthiness. This project aims to address this issue by evaluating machine learning algorithms for classifying images as either AI-generated or real.

The goal is to enhance the authenticity verification process for images by comparing SVM, KNN, and CNN algorithms. We utilized a dataset from Kaggle for this study, focusing on a subset of images due to resource constraints.

## Dataset Description

Kaggle Dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

The dataset used includes two classes—REAL and FAKE:
- **REAL**: Images from the CIFAR-10 dataset.
- **FAKE**: Images generated using Stable Diffusion version 1.4.

The dataset comprises 100,000 images for training (50k per class) and 20,000 for testing (10k per class). Due to resource limitations, we used a subset: 5,000 real images and 5,000 fake images for training, and 1,500 real images and 1,500 fake images for testing.

### Pre-processing

Images were resized to 32x32 pixels and converted to RGB format. Data augmentation techniques, such as flipping, were applied to increase variability in the training data.

## Machine Learning Algorithms

### 1. Support Vector Machine (SVM)

SVM constructs hyper-planes that maximize the margin between classes. The RBF kernel was used, and the best parameters were found to be C=10 and gamma=0.0001. Accuracy achieved was 79.0% on the full dataset.

### 2. K-Nearest Neighbors (KNN)

KNN classifies based on majority voting from the k-nearest neighbors. The optimal parameters were k=31, uniform weights, and Manhattan distance. Accuracy achieved was 69.03%, with a confusion matrix highlighting precision at 72.78%.

### 3. Convolutional Neural Network (CNN)

CNNs utilize convolutional layers to extract features and reduce dimensionality. The network used had two convolutional layers and a dense output layer. With 5 epochs, the CNN achieved an accuracy of approximately 93.56%.

## Results

The CNN emerged as the most effective method for detecting AI-generated images, demonstrating the highest accuracy and efficient training time compared to SVM and KNN. 

## Future Directions

Future improvements could involve using the complete dataset, employing better hardware, and optimizing preprocessing techniques. Exploring GPU utilization could also enhance performance. Additionally, real-time detection systems could be developed to verify image authenticity online.

## Conclusion

This project successfully evaluated and compared SVM, KNN, and CNN for AI-generated image detection. CNN proved to be the most effective method with the highest accuracy. This highlights its suitability for image classification tasks involving real versus AI-generated images. The results can be replicated by running any of the three Jupyter notebooks.

