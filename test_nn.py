import time
import random as rand
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

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

for _ in range(10):
    train, test = train_test_split(data_set, test_size=0.2)

    nn.train_on_batch(train[:, 1:10], train[0])

    loss = nn.predict_on_batch(test[1:10])





