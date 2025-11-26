🐶🐱 Dog vs Cat Image Classification — CNN Project

This project aims to classify images of dogs and cats using a Convolutional Neural Network (CNN) built completely from scratch (without transfer learning).
It demonstrates deep learning fundamentals such as data preprocessing, model design, training, evaluation, and prediction.

📌 Overview

Build a custom CNN for binary image classification

Train on 8000+ images of cats and dogs

Use data augmentation to improve generalization

Apply dropout to reduce overfitting

Evaluate model using accuracy and confusion matrix

Make predictions on new unseen images

📂 Dataset

The dataset contains two categories:

cats

dogs

Images are divided into train, validation, and test sets.
The dataset is sourced from Kaggle: Dog and Cat Classification Dataset.

🧠 Model Description

The CNN includes:

Multiple convolution layers for feature extraction

ReLU activation for non-linearity

Max-pooling layers for spatial downsampling

Fully connected dense layers for classification

Dropout layer to reduce overfitting

Softmax output layer for 2-class prediction

🛠 Tools & Technologies

Python

TensorFlow / Keras

NumPy

Matplotlib

scikit-learn

KaggleHub (for dataset download)

🎯 Training

The model is trained using:

Adam Optimizer

Categorical Crossentropy Loss

Mini-batch gradient descent

Data augmentation (rotation, zoom, flip, shift)

Early stopping to prevent over-training

📊 Evaluation

The project includes:

Training accuracy

Validation accuracy

Loss curves

Confusion matrix



🔍 Prediction

The project includes functionality to test the model on:

Individual cat/dog images

Entire folders of images

The model outputs whether the image is predicted as Dog or Cat.

📦 Project Structure
project-folder/
├── data/
├── models/
├── notebooks/
├── src/
├── README.md
└── requirements.txt
