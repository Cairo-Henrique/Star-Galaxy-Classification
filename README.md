# Star-Galaxy-Classification

This project aims to classify astronomical images as either stars or galaxies using a Convolutional Neural Network (CNN) implemented in PyTorch, as well as some other models for comparition.

## Project Description

The project involves the following steps:

1. **Data Acquisition:** Download and extract the dataset containing images of stars and galaxies.
2. **Data Preparation:** Load, process, and augment the image data for training the model.
3. **Model Definition:** Define the CNN architecture using PyTorch.
4. **Training:** Train the CNN model on the prepared dataset.
5. **Evaluation:** Evaluate the model's performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.
6. **Misclassification Analysis:** Identify and visualize misclassified images to understand the model's limitations.

## Dataset

The dataset used in this project is obtained from Kaggle (https://www.kaggle.com/datasets/divyansh22/dummy-astronomy-data). It contains a collection of star and galaxy images in JPG format.

## Dependencies

The project requires the following libraries:

- Python 3.x
- torch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- Pillow
- gdown

## Results

Accuracy: 90%
