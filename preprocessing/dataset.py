import json

from tensorflow import keras
import numpy as np
import os


class Dataset():
    def __init__(self, train_path, val_path, param):
        self.train_path = train_path
        self.validation_path = val_path

        self.height = param['Height']
        self.width = param['Width']
        self.channels = param['Channels']
        self.batch_size = param['BatchSize']
        self.classes = param['NumClass']

        self.x_train, self.y_train = self.build_set()
        self.x_val, self.y_val = self.build_set(mode='validation')

        self.x_train = self.x_train/255.0
        self.x_val = self.x_val/255.0
        self.hot_encoding()

    def hot_encoding(self):
        self.y_train = keras.utils.to_categorical(self.y_train, num_classes=self.classes)
        self.y_val = keras.utils.to_categorical(self.y_val, num_classes=self.classes)

    def build_set(self, mode='training'):
        x = []
        y = []

        if mode == 'training':
            fold_path = self.train_path
        else:
            fold_path = self.validation_path

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
            for image in os.listdir(path):
                img_file = keras.preprocessing.image.load_img(path + '/' + image, target_size=(self.height, self.width))
                img_arr = np.asarray(img_file)
                x.append(img_arr)
                y.append(label)

        x = np.asarray(x)
        y = np.asarray(y)

        return x, y

    def get_training(self):
        return self.x_train, self.y_train

    def get_validation(self):
        return self.x_val, self.y_val

    def get_train_flow(self):
        return self.train_flow


if __name__ == "__main__":
    json_file = open('config/config.json', 'r')
    json_file = json.load(json_file)

    training_path = json_file['Training']
    validation_path = json_file['Validation']
    parameters = json_file['Parameters']

    dataset = Dataset(training_path, validation_path, parameters)
    training = dataset.get_training()
    validation = dataset.get_validation()
