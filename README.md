# Potato Leaf Disease Detection Using Neural Networks

This project explores the detection of potato leaf diseases using image-based deep learning. The goal is to identify healthy potato leaves and distinguish between Early Blight and Late Blight from leaf images.

## Project Overview

Potato leaf disease detection is important for identifying plant health problems early and supporting better crop management. Disease symptoms can vary by crop condition and environmental factors, making visual classification challenging.

The broader research described in the original project context developed a multi-level deep learning approach. It first extracts potato leaves from plant images using YOLOv5-based segmentation, then applies a convolutional neural network to recognize Early Blight and Late Blight from the leaf images.

This repository contains a Jupyter notebook that prepares potato leaf images, trains a convolutional neural network, and evaluates the model's ability to classify three categories:

- Early Blight
- Late Blight
- Healthy

## Dataset

The working dataset is organized into Training, Testing, and Validation folders. The notebook combines 4,072 available images before creating a reproducible training and testing split.

The images are:

- Grouped by disease or healthy class.
- Resized for consistent model input.
- Split into 80% training data and 20% testing data.
- Used for training and held-out evaluation.

The original README also references a potato leaf disease dataset of 4,062 images collected from the Central Punjab region of Pakistan, along with evaluation on the PlantVillage dataset.

Early approaches for potato leaf disease detection can be limited when models are trained and tested on images from a specific region or set of growing conditions. The proposed research approach was therefore designed to detect disease from leaf images rather than relying only on the wider plant context.

## Model and Results

A convolutional neural network is used to classify the leaf images. The current implementation achieved:

- Training accuracy: 82.76%
- Validation accuracy: 78.99% in the final epoch
- Best validation accuracy: 83.13%
- Held-out test accuracy: 76.81%

The test results indicate that the model performs particularly well for Late Blight and Early Blight samples. Healthy leaf classification has high precision but lower recall, meaning that some Healthy samples are still being missed or confused with another class.

The referenced research approach reported 99.75% accuracy on the Central Punjab potato leaf disease dataset and was also evaluated on the PlantVillage dataset. It was compared with state-of-the-art models and achieved strong accuracy and computational efficiency.

## Repository Contents

- `pld.ipynb` - Data preparation, model development, training, and evaluation notebook.
- `Project-Flow.md` - Detailed summary of the notebook workflow and recorded results.
- `dataset/` - Training, testing, and validation plant leaf images.
- `research_paper/` - Reference material about plant leaf disease detection and classification.
