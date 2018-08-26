from tensorflow.keras import *
from tensorflow.keras.layers import *

# CONSIDERARE QUESTA COME CUSTOM 1
def custom_cnn_2(num_classes, heigth, width, channels):
    """
    :param num_classes: Number of classes to classify
    :param heigth: Height of images
    :param width: Width of images
    :param channels: The number of channels of images (3 if rgb)
    :return: model
    """

    input_shape = (heigth, width, channels)

    model = Sequential()

    model.add(
        Conv2D(input_shape=input_shape, filters=16, kernel_size=(5, 5), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))  # Regularization

    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.2))  # Regularization

    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(0.2))  # Regularization

    model.add(Dense(units=num_classes, activation='softmax'))

    return model