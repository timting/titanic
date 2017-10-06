import time
import random as rand
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

class QNN(object):
    def __init__(self, \
        alpha = 0.2, \
        gamma = 0.99, \
        rar = 0.7, \
        num_actions = 2, \
        num_features = 10, \
        radr = 0.9999):

        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.state = None
        self.action = None
        self.start = 0
        self.num_actions = num_actions
        self.num_features = num_features
        self.replay_memory = np.array([])
        self.good_memory = np.array([])
        self.sgd = optimizers.SGD(lr=self.alpha, decay=1e-6, momentum=0.4, nesterov=False)

        self.nn = Sequential()
        self.nn.add(Dense(64, input_dim=8))
        self.nn.add(Activation('relu'))
        self.nn.add(Dense(64))
        self.nn.add(Activation('relu'))
        self.nn.add(Dense(4))
        self.nn.add(Activation('linear'))
        self.nn.compile(loss='mean_squared_error', optimizer=self.sgd)

    def set_initial_state(self, state):
        self.state = state
        self.action = 2

        return self.action

    def update(self, state_prime, reward, done):
        tuples = []

        # If memory is large enough, start sampling from it
        if len(self.replay_memory) > 200:
            indices = rand.sample(range(len(self.replay_memory) - 1), 64)
            tuples = self.replay_memory[indices, :]

        if len(self.good_memory) > 5:
            indices = rand.sample(range(len(self.good_memory) - 1), 3)
            good_tuples = self.good_memory[indices, :]
            tuples = np.append(tuples, good_tuples, axis=0)

        if tuples == []:
            tuples.insert(0, [self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.action, reward, state_prime[0], state_prime[1], state_prime[2], state_prime[3], state_prime[4], state_prime[5], state_prime[6], state_prime[7]])
            tuples = np.array(tuples)
        else:
            tuples = np.insert(tuples, 0, [self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.action, reward, state_prime[0], state_prime[1], state_prime[2], state_prime[3], state_prime[4], state_prime[5], state_prime[6], state_prime[7]], axis=0)

        chosen_action = self.batch_update(tuples, done)

        # Add tuple to memory (once we're full, do a ring update of the array)
        if len(self.replay_memory) == 0:
           self.replay_memory = np.array([[self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.action, reward, state_prime[0], state_prime[1], state_prime[2], state_prime[3], state_prime[4], state_prime[5], state_prime[6], state_prime[7]]])
        elif len(self.replay_memory) > 100000:
            self.replay_memory[self.start] = [self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.action, reward, state_prime[0], state_prime[1], state_prime[2], state_prime[3], state_prime[4], state_prime[5], state_prime[6], state_prime[7]]
            self.start = (self.start + 1) % len(self.replay_memory)
        else:
            self.replay_memory = np.append(self.replay_memory, [[self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.action, reward, state_prime[0], state_prime[1], state_prime[2], state_prime[3], state_prime[4], state_prime[5], state_prime[6], state_prime[7]]], axis=0)

        # Add tuple to good memory if reward is 100
        if reward == 100 and len(self.good_memory) == 0:
            self.good_memory = np.array([[self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.action, reward, state_prime[0], state_prime[1], state_prime[2], state_prime[3], state_prime[4], state_prime[5], state_prime[6], state_prime[7]]])
        elif reward == 100:
            self.good_memory = np.append(self.good_memory, [[self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.action, reward, state_prime[0], state_prime[1], state_prime[2], state_prime[3], state_prime[4], state_prime[5], state_prime[6], state_prime[7]]], axis=0)

        self.state = state_prime

        if rand.uniform(0.0, 1.0) < self.rar:
            chosen_action = rand.randint(0, self.num_actions-1)

        self.rar = self.rar * self.radr
        if self.rar < 0.25 and len(self.good_memory) <= 5:
            self.rar = 0.25
        # less exploration if we have more than 5 good landings
        elif self.rar < 0.05 and len(self.good_memory) > 5:
            self.rar = 0.05

        self.action = chosen_action

        return chosen_action

    # First of the batch should always be the real sars' tuple.
    def batch_update(self, tuples, done):
        number_of_tuples = len(tuples)
        actions = tuples[:, 8]
        rewards = tuples[:, 9]
        predict_on = np.concatenate((tuples[:, 0:8], tuples[:, 10:18]), axis=0)
        predictions = self.nn.predict_on_batch(predict_on)
        state_predictions = predictions[0:number_of_tuples, :]
        state_prime_predictions = predictions[number_of_tuples:(number_of_tuples*2), :]
        chosen_actions = np.argmax(state_prime_predictions, axis=1)
        next_q_values = np.max(state_prime_predictions, axis=1)

        if done == True:
            print state_predictions[0]
            print rewards[0]

        q_value_targets = state_predictions
        for i in range(len(state_predictions)):
            # if i == 0:
            #     print "-------------"
            #     print rewards[i]
            #     print next_q_values[i]
            #     print "Old q value:"
            #     print q_value_targets[i, int(actions[i])]

            # If reward is -100 or 100, we're done and next_q_value should be 0
            # Check this for all states, even replay, so they don't throw off our q values
            if rewards[i] == 100 or rewards[i] == -100:
                next_q_values[i] = 0

            q_value_targets[i, int(actions[i])] = rewards[i] + self.gamma * next_q_values[i]
            # if i == 0:
            #     print "New q value:"
            #     print q_value_targets[i, int(actions[i])]

        self.nn.train_on_batch(tuples[:, 0:8], np.array(q_value_targets))

        return chosen_actions[0]

    # Test the weights given - don't update
    def select_action(self, state_prime, reward, done):
        prediction = self.nn.predict_on_batch(np.array([state_prime]))
        chosen_action = np.argmax(prediction[0])

        return chosen_action

    def get_weights(self):
        return self.nn.get_weights()

    def save_weights(self):
        np.save("weights.npy", self.get_weights())

    def set_weights(self, weights):
        self.nn.set_weights(np.array(weights))
