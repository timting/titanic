import time
import random as rand
import numpy as np
import pandas as pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

sgd = optimizers.SGD(lr=0.2, decay=1e-6, momentum=0.4, nesterov=False)
nn = Sequential()
nn.add(Dense(64, input_dim=10))
nn.add(Activation('relu'))
nn.add(Dense(64))
nn.add(Activation('relu'))
nn.add(Dense(2))
nn.add(Activation('linear'))
nn.compile(loss='mean_squared_error', optimizer=sgd)

training_set = pandas.read_csv('train.csv').as_matrix()
print training_set


#nn.train_on_batch(tuples[:, 0:8], np.array(q_value_targets))

#nn.predict_on_batch(predict_on)

# calculate accuracy