from tensorflow.keras import *
from tensorflow.keras.layers import *


def lenet_5(num_classes, heigth, width, channels):
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
        Conv2D(input_shape=input_shape, filters=6, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model
