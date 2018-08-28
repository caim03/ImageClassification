from tensorflow import keras
from preprocessing.pixel_intensity import plot_pixel_intensity
import numpy as np
import os

''' This class is used to load the images from disk and create three different set: training set, validation set and test set 
    Every set is composed by the list of images with their labels in one-hot encoding '''


class Dataset:
    def __init__(self, train_path, val_path, test_path, param):
        self.train_path = train_path
        self.validation_path = val_path
        self.test_path = test_path

        self.height = param['Height']
        self.width = param['Width']
        self.channels = param['Channels']
        self.batch_size = param['BatchSize']
        self.classes = param['NumClass']

        self.labels_counter = []

        self.x_train, self.y_train = self.build_set()
        self.x_val, self.y_val = self.build_set(mode='validation')  # Take val images
        self.x_test, self.y_test = self.build_set(mode='test')  # Take test images

        # Verify if data must be normalized
        # Decomment this line to plot pixel intensity before normalization
        # plot_pixel_intensity(self.x_train[0])

        mean = self.x_train.mean(axis=0)  # Column mean
        std = self.x_train.std(axis=0)  # Column std

        # All sets are normalized with same values
        self.x_train = (self.x_train - mean)/std
        self.x_val = (self.x_val - mean)/std
        self.x_test = (self.x_test - mean)/std

        self.hot_encoding()

    def hot_encoding(self):
        """
        This function transforms the labels in one-hot encoding way. E.g class 3 is [0,0,1,0] in one-hot encoding
        :return: None
        """
        self.y_train = keras.utils.to_categorical(self.y_train, num_classes=self.classes)
        self.y_val = keras.utils.to_categorical(self.y_val, num_classes=self.classes)
        self.y_test = keras.utils.to_categorical(self.y_test, num_classes=self.classes)

    def build_set(self, mode='training'):
        """
        This function builds the set defined into mode parameter
        :param mode: This parameter is used to discriminate the correct folder from which get data
        :return: None
        """
        x = []
        y = []

        if mode == 'training':
            fold_path = self.train_path
        elif mode == 'validation':
            fold_path = self.validation_path
        else:
            fold_path = self.test_path

        for folder in os.listdir(fold_path):
            if folder == 'NEUTROPHIL':
                label = 0
            elif folder == 'LYMPHOCYTE':
                label = 1
            elif folder == 'MONOCYTE':
                label = 2
            else:
                label = 3

            path = fold_path + '/' + folder
            count = 0  # This variable is used to count the number of images within every class
            for image in os.listdir(path):
                if mode == 'training':
                    count = count + 1
                img_file = keras.preprocessing.image.load_img(path + '/' + image, target_size=(self.height, self.width))
                img_arr = np.asarray(img_file)
                x.append(img_arr)
                y.append(label)

            if mode == 'training':
                self.labels_counter.append(count)

        x = np.asarray(x)
        y = np.asarray(y)

        return x, y

    def get_training(self):
        """
        This function returns the training set
        :return: None
        """
        return self.x_train, self.y_train

    def get_validation(self):
        """
        This function returns the validation set
        :return: None
        """
        return self.x_val, self.y_val

    def get_test(self):
        """
         This function returns the test set
        :return: None
                """
        return self.x_test, self.y_test

    def get_counters(self):
        """
        This function returns a list of counters. Every counter indicates the number of images within a specific class
        :return: None
        """
        return self.labels_counter
