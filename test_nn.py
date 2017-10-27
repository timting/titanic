import time
import random as rand
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """

    confusion_matrix = [[0, 0],[0, 0]]
    for i in range(len(classifier_output)):
        if classifier_output[i] == 0 and true_labels[i] == 0:
            confusion_matrix[1][1] += 1
        elif classifier_output[i] == 1 and true_labels[i] == 0:
            confusion_matrix[1][0] += 1
        elif classifier_output[i] == 0 and true_labels[i] == 1:
            confusion_matrix[0][1] += 1
        elif classifier_output[i] == 1 and true_labels[i] == 1:
            confusion_matrix[0][0] += 1

    return confusion_matrix

def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """

    cm = confusion_matrix(classifier_output, true_labels)

    return (cm[0][0] + cm[1][1]) / float(sum(cm[0]) + sum(cm[1]))


sgd = optimizers.SGD(lr=0.2, decay=1e-6, momentum=0.4, nesterov=False)
nn = Sequential()
nn.add(Dense(64, input_dim=9))
nn.add(Activation('relu'))
nn.add(Dense(64))
nn.add(Activation('relu'))
nn.add(Dense(2))
nn.add(Activation('linear'))
nn.compile(loss='mean_squared_error', optimizer=sgd)

data_set = pd.read_csv('normalized-train.csv').as_matrix()
rows = data_set.shape[0]

train, test = train_test_split(data_set, test_size=0.2)
class_0 = (train[:, 0] == 0).astype(int)
class_1 = (train[:, 0] == 1).astype(int)
class_0 = np.transpose(np.matrix(class_0))
class_1 = np.transpose(np.matrix(class_1))
classes = np.concatenate((class_0, class_1), axis=1)

for _ in range(100):
    nn.train_on_batch(train[:, 1:10], classes)

    loss = nn.predict_on_batch(test[:, 1:10])

    predictions = np.argmax(loss, axis=1)

    print(accuracy(predictions, test[:, 0]))


