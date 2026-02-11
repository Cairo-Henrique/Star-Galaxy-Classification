# Star-Galaxy-Classification

This project aims to classify astronomical images as either stars or galaxies using a Convolutional Neural Network (CNN) implemented in PyTorch, as well as some other models for comparision.

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

### Other models
Accuracy: ~77% 

The SVC, MLP, and Random Forest models failed to effectively extract discriminative information from the engineered features. The extremely low F1-scores for the galaxy class indicate that these models were strongly biased toward predicting the majority class (star), effectively classifying most samples as stars. This behavior is consistent with the class imbalance present in the dataset.

### CNN
Accuracy: 90%

F1 Score: 80%

The convolutional layers were able to capture the relevant spatial features directly from the images, leading to a significant performance improvement—especially after applying data augmentation.
An analysis of the misclassified samples shows that the model struggles mainly with images containing more than one celestial object. This limitation is expected, as such cases represent a small fraction of the dataset and are therefore underrepresented during training.
