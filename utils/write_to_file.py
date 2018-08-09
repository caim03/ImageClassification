from sklearn import metrics
import numpy as np


def write_to_file(score, y_val, y_pred, dict_elem, batch_size, epochs, model_name, name='Results.txt', mode='w+'):
    f = open(name, mode)

    f.write('This summary is generated with a batch size of: ' + str(batch_size) + ' and a number of epochs set to: ' + str(epochs) + '\n\n')
    f.write(model_name + ' Model - Evaluate Loss:' + str(score[0]) + '\n')
    f.write(model_name + ' Model - Evaluate Accuracy:' + str(score[1]) + '\n\n')
    f.write(metrics.classification_report(np.where(y_val > 0)[1], np.argmax(y_pred, axis=1),
                                          target_names=list(dict_elem.values())))

    f.close()