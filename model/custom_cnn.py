from tensorflow.keras import *
from tensorflow.keras.layers import *


def cnn(num_classes, heigth, width, channels):
    """

    :param num_classes: Number of classes to classify
    :param heigth: Height of images
    :param width: Width of images
    :param channels: The number of channels of images (3 if rgb)
    :return: model
    """

    input_shape = (heigth, width, channels)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, strides=1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
