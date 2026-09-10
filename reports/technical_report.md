# Technical Report — AI-Powered Image Classification Platform

## Objective
Build and deploy an image classification platform using transfer learning.

## Dataset
CIFAR-10: 60,000 32×32 RGB images across 10 object classes.

## Data pipeline
Images are resized to 224×224 and normalized with ImageNet statistics. Training data uses random horizontal flips, random crops, and color jitter. The official training set is split into training and validation partitions; the official test set is held out for final evaluation.

## Models compared
- ResNet18 with ImageNet transfer learning
- MobileNetV3-Small with ImageNet transfer learning

The convolutional backbones are initially frozen and classification heads are fine-tuned.

## Performance comparison

### resnet18
Accuracy: 0.7790
Precision: 0.7876
Recall: 0.7790
F1: 0.7748
Top-5 Accuracy: 0.9835

### mobilenet_v3_small
Accuracy: 0.8065
Precision: 0.8158
Recall: 0.8065
F1: 0.8063
Top-5 Accuracy: 0.9945

## Selected model
**mobilenet_v3_small** was selected using highest test accuracy among the compared trained candidates.
