import json
from sklearn import metrics

from tensorflow import keras
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from preprocessing.dataset import Dataset
from model.custom_cnn import cnn
from model.metrics_saver import Metrics, plot_learning_curve, plot_confusion_matrix
from matplotlib import pyplot as plt
import numpy as np

dict_elem = {1: 'NEUTROPHIL', 2: 'LYMPHOCYTE', 3: 'MONOCYTE', 4: 'EOSINOPHIL'}


def main():
    json_file = open('config/config.json', 'r')
    json_file = json.load(json_file)

    training_path = json_file['Training']
    validation_path = json_file['Validation']
    parameters = json_file['Parameters']

    dataset = Dataset(training_path, validation_path, parameters)

    x_train, y_train = dataset.get_training()
    x_val, y_val = dataset.get_validation()

    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    train_batches = len(x_train) // parameters['BatchSize']  # // operator indicates a floor division
    val_batches = len(x_val) // parameters['BatchSize']

    model = cnn(num_classes=parameters['NumClass'], heigth=parameters['Height'], width=parameters['Width'],
                channels=parameters['Channels'])
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(),
                  metrics=['accuracy'])

    history = model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=parameters['BatchSize']),
                                  steps_per_epoch=train_batches, validation_data=[x_val, y_val],
                                  epochs=parameters['NumEpoch'], callbacks=[Metrics('logs')])

    score = model.evaluate(x=x_val, y=y_val, verbose=0)

    print('Custom CNN Model #1C - Loss: \n', score[0])
    print('Custom CNN Model #1C - Accuracy: \n', score[1])

    y_pred = model.predict(x_val)

    print('\n')
    print(metrics.classification_report(np.where(y_val > 0)[1], np.argmax(y_pred, axis=1),
                                        target_names=list(dict_elem.values())))
    print('\n')

    classes_predicted = np.argmax(y_pred, axis=1)
    classes_true = np.argmax(y_val, axis=1)

    plot_learning_curve(history)
    plt.show()

    confusion_mtx = metrics.confusion_matrix(classes_true, classes_predicted)
    plot_confusion_matrix(confusion_mtx, classes=list(dict_elem.values()))
    plt.show()


if __name__ == "__main__":
    main()
