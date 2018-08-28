import json
from sklearn import metrics

from tensorflow import keras
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from preprocessing.dataset import Dataset
from preprocessing.pixel_intensity import plot_pixel_intensity
from preprocessing.balance import balance
from model.custom_cnn import cnn
from model.alex_net import alex_net
from model.lenet_5 import lenet_5
from model.custom_cnn2 import custom_cnn_2
from model.vgg16 import vgg16
from model.metrics_saver import Metrics, plot_learning_curve, plot_confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from utils.write_to_file import write_to_file
from tensorflow.keras.callbacks import ModelCheckpoint

dict_elem = {1: 'NEUTROPHIL', 2: 'LYMPHOCYTE', 3: 'MONOCYTE', 4: 'EOSINOPHIL'}


def main():
    # TODO hyperopt for optimal parameter

    # --- OPEN CONFIGURATION FILE AND GET PARAMETERS---
    json_file = open('config/config.json', 'r')
    json_file = json.load(json_file)

    training_path = json_file['Training']
    validation_path = json_file['Validation']
    parameters = json_file['Parameters']

    # --- CREATE DATASET AND COMPUTE A PRE-PROCESSING---
    dataset = Dataset(training_path, validation_path, parameters)
    x_train, y_train = dataset.get_training()
    x_val, y_val = dataset.get_validation()

    # Decomment this lines to performs a pre-processing study
    # After normalization
    #plot_pixel_intensity(x_train[0], './pixel_intensity_after_normalization.png')

    # Verify if dataset is balanced
    #counters = dataset.get_counters()
    #balance(counters)

    # Data augmentation in real-time to reduce overfitting (regularization)
    datagen = keras.preprocessing.image.ImageDataGenerator()
        #rotation_range=10,  # randomly rotate images in the range
        #width_shift_range=0.1,  # randomly shift images horizontally
        #height_shift_range=0.1,  # randomly shift images vertically
        #horizontal_flip=True)  # randomly flip images

    train_batches = len(x_train) // parameters['BatchSize']  # // operator indicates a floor division

    # --- DEFINE MODEL ---
    model = custom_cnn_2(num_classes=parameters['NumClass'], heigth=parameters['Height'], width=parameters['Width'],
                channels=parameters['Channels'])

    lr = parameters['LearningRate']
    decay = lr/parameters['NumEpoch']

    # --- TRAIN MODEL ---
    model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=lr, decay=decay),
                  metrics=['accuracy'])

    # --- CHECKPOINTING TO SAVE BEST NETWORK ---
    filepath = 'models_saved/custom_cnn_2.hdf5'
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    history = model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=parameters['BatchSize']),
                                  steps_per_epoch=train_batches, validation_data=[x_val, y_val],
                                  epochs=parameters['NumEpoch'], callbacks=[Metrics('logs'), checkpoint])

    model.load_weights('models_saved/custom_cnn_2.hdf5')
    model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=lr, decay=decay),
                  metrics=['accuracy'])

    # --- EVALUATE MODEL ---
    score = model.evaluate(x=x_val, y=y_val, verbose=0)

    # --- PREDICT NEW VALUES ---
    y_pred = model.predict(x_val)  # Use validation because in this way we can evaluate some metrics with sklearn

    write_to_file(score, y_val, y_pred, dict_elem, parameters['BatchSize'], parameters['NumEpoch'], "Custom CNN 2")

    classes_predicted = np.argmax(y_pred, axis=1)
    classes_true = np.argmax(y_val, axis=1)

    # --- PLOT RESULTS ---
    plot_learning_curve(history)
    plt.show()

    confusion_mtx = metrics.confusion_matrix(classes_true, classes_predicted)
    plot_confusion_matrix(confusion_mtx, classes=list(dict_elem.values()))
    plt.show()

    # --- SAVE MODEL AND WEIGHTS ---
    model_json = model.to_json()
    with open('models_saved/custom_cnn_2.json', 'w') as mod:
        mod.write(model_json)

    print("Model was saved successfully!")

if __name__ == "__main__":
    main()
