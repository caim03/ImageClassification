from tensorflow.keras import *
from tensorflow.keras.layers import *


def alex_net(num_classes, heigth, width, channels):
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
        Conv2D(input_shape=input_shape, filters=32, kernel_size=(5, 5), strides=1, padding='same'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(units=256))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Activation('relu'))

    model.add(Dropout(0.5))  # Regularization

    model.add(Dense(units=256))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Activation('relu'))

    model.add(Dropout(0.5))  # Regularization

    model.add(Dense(units=num_classes))
    model.add(BatchNormalization())  # Add a Batch Normalization stage between linear and non linear layer
    model.add(Activation('softmax'))

    return model