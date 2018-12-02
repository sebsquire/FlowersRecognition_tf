# FlowersRecognition_tf
Multi-class Image Classification using CNN (Flower Types) (Tensorflow, OpenCV)
Dataset available at: https://www.kaggle.com/alxmamaev/flowers-recognition

Includes:
 - Tensorflow CNN implemented via functions
 - Tensorflow CNN implemented via classes
 - Dropout in fully connected layer to reduce overfit
 - Normalisation of image data

Modules:
 - prepocessing: Function for creation of training and test data in .npy files.
 - Main_functions: calls preprocessing.py to load and preprocess image data, sets model parameters, and trains, tests, and produces classification metrics for the model on the validation set using functions.
 - Main_classes: calls preprocessing.py to load and preprocess image data, sets model parameters, and trains, tests, and produces classification metrics for the model on the validation set via implementation of ConvNNModel class.

CNN model:
  - 5 convolutional layers w/ max pooling proceeding each of the first 4.
  - 1 fully connected layer
  - 1 output layer (softmax)
Model attains 71% accuracy in 30 epochs using:
  - AdamOptimizer with lr=0.001, weight_decay=1e-6, keep_rate=0.95, batch_size=16

Python 3.6.7, Tensorflow 1.11.0
