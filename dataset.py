from __future__ import print_function

import tensorflow as tf
import os
import json


class Dataset():
    def __init__(self, path, param):
        self.eos_path = path['Eos']
        self.lym_path = path['Lym']
        self.mon_path = path['Mon']
        self.neu_path = path['Neu']

        self.height = param['Height']
        self.width = param['Width']
        self.channels = param['Channels']
        self.batch_size = param['BatchSize']
        self.prefetch_size = param['PrefetchSize']
        self.num_epoch = param['NumEpoch']

        self.lenght = None
        self.iterator = None
        self.set = None

        self.filenames = []
        self.labels = []

        self.build_dataset()

    def build_dataset(self):
        self.read_paths()

        # Parse training set images
        set = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        set = set.shuffle(len(self.filenames))
        set = set.map(self._parse_function, num_parallel_calls=2)
        set = set.batch(self.batch_size)
        set = set.prefetch(self.prefetch_size)
        set = set.repeat(self.num_epoch)
        self.set = set
        self.iterator = tf.data.Iterator.from_structure(set.output_types, set.output_shapes)

    def read_paths(self):
        eos = [self.eos_path + i for i in os.listdir(self.eos_path)]
        lym = [self.lym_path + i for i in os.listdir(self.lym_path)]
        mon = [self.mon_path + i for i in os.listdir(self.mon_path)]
        neu = [self.neu_path + i for i in os.listdir(self.neu_path)]

        # Set labels

        label_eos = ["eosinophil" for i in eos]
        label_lym = ["lymphocyte" for i in lym]
        label_mon = ["monocyte" for i in mon]
        label_neu = ["neutrophil" for i in neu]

        # Merge all

        self.filenames = eos + lym + mon + neu
        self.labels = label_eos + label_lym + label_mon + label_neu

        for i in range(len(self.labels)):
            if self.labels[i] == 'eosinophil':
                self.labels[i] = 0
            elif self.labels[i] == 'lymphocyte':
                self.labels[i] = 1
            elif self.labels[i] == 'monocyte':
                self.labels[i] = 2
            else:
                self.labels[i] = 3

        self.lenght = len(self.filenames)

    def init_iterator(self):
        return self.iterator.make_initializer(self.set)

    def get_next(self):
        return self.iterator.get_next()

    def _parse_function(self, filename, label):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=self.channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.height, self.width])

        #label = tf.one_hot(label, 4)
        return image, label
