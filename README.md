# FlowersRecognition_tf
Multi-class Image Classification using CNN (Flower Types) (Tensorflow, OpenCV)
Dataset available at: https://www.kaggle.com/alxmamaev/flowers-recognition

Includes:
 - Tensorflow CNN implemented via functions
 - Tensorflow CNN implemented via classes
 - Normalisation of image data

Modules:
 - prepocessing: Function for creation of training and test data in .npy files.
 - Main_functions: calls preprocessing.py to load and preprocess image data, sets model parameters, and trains, tests, and produces classification metrics for the model on the validation set using functions.
 - Main_classes: calls preprocessing.py to load and preprocess image data, sets model parameters, and trains, tests, and produces classification metrics for the model on the validation set via implementation of ConvNNModel class.
