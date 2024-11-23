# Project Overview
This project implements a Convolutional Neural Network (CNN) to classify hand gestures from grayscale images into 10 categories. The model is trained on a dataset of pre-classified images of hand gestures and achieves a high accuracy on both training and testing data.

## Dataset
Categories:
Palm
L
Fist
Fist Moved
Thumb
Index
OK
Palm Moved
C
Down

## Data Preprocessing:
Images resized to 50x50 pixels.
Grayscale images were used for uniformity.
Normalized pixel values to range [0, 1].

## Environment Setup
Dependencies:
Python 3.7+
TensorFlow 2.x
Keras
OpenCV
NumPy
Matplotlib
Seaborn

## Model Architecture:
### Convolutional Layers:
First Conv2D layer: 32 filters, kernel size (3x3), ReLU activation.
Second Conv2D layer: 32 filters, kernel size (3x3), ReLU activation, followed by MaxPooling and Dropout.
Third Conv2D layer: 64 filters, kernel size (3x3), ReLU activation, followed by MaxPooling and Dropout.

Fully Connected Layers:
Flatten layer to reshape data.
Dense layer with 256 neurons and ReLU activation.
Dense output layer with 10 neurons and softmax activation for classification.

Dropout:
Added after pooling layers to reduce overfitting.
Total Parameters: 1,669,290
Training Details

Optimizer: RMSprop

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Batch Size: 32

Epochs: 7

Train-Test Split: 70%-30%

## Performance

Training Accuracy: ~99.89%
Validation Accuracy: ~99.95%
Test Loss: 0.0051
Test Accuracy: 99.95%

## Results Visualization
Loss vs. Epochs:
Accuracy vs. Epochs:
Confusion Matrix:
A heatmap visualization of the confusion matrix is generated.
