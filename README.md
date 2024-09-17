# CNN-based-plant-disease-detection-

Rose Disease Classification using Convolutional Neural Networks (CNN)
This project involves building a Convolutional Neural Network (CNN) model to classify rose plant diseases based on image data. Using TensorFlow Keras, OpenCV, and image processing techniques, the model can identify different disease types in rose plants. The project applies deep learning for image classification, leveraging data augmentation, model optimization, and early stopping to improve performance.

#Project Overview
This project focuses on developing a deep learning model for the classification of rose plant diseases using a CNN architecture. The model is trained on image data of rose leaves, applying data augmentation techniques to improve generalization and performance.

#Key Features:
Image Preprocessing: Load, resize, and normalize input images for the CNN.
Data Augmentation: Rotation, shifting, shearing, zooming, and flipping to improve generalization.
CNN Model: Sequential model architecture with convolutional, pooling, dense, and dropout layers.
Early Stopping: Monitors validation loss to prevent overfitting.
Accuracy Evaluation: Evaluate the trained model's performance using accuracy metrics.

#Model Architecture
The CNN model is built using the Sequential API in Keras with the following layers:
Conv2D: Extracts features from the input images.
MaxPooling2D: Reduces spatial dimensions, keeping essential features.
Flatten: Converts the 2D matrix data to a 1D vector.
Dense: Fully connected layers for classification.
Dropout: Helps reduce overfitting by randomly dropping neurons during training.

#Data Preprocessing
Images are loaded using OpenCV (cv2), resized to 128x128, and normalized to a pixel range between 0 and 1.
Data is split into training and testing sets using train_test_split.
Data Augmentation is applied to the training set to improve generalization. Augmentations include:
Rotation, Width/Height shift, Shear, Zoom, and Horizontal Flip.

#Training the Model
The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.
An early stopping callback is added to halt training when validation loss does not improve for a defined number of epochs.

#Evaluation and Prediction
The trained model is evaluated on the test set using accuracy metrics. Additionally, the model is used to predict the disease class of a new input image after preprocessing.

>To make predictions on new images:
Load the image using OpenCV.
Preprocess the image similarly to training images.
Use model.predict() to predict the disease class.
