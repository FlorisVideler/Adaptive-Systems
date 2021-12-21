# import tensorflow as tf
# from keras import Sequential
# from keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

import tensorflow.compat.v1 as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam


class FunctionApproximator:
    """The neural network class, used for the policy and the target network"""
    def __init__(self, n_states, n_actions):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=n_states, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(n_actions, activation="linear"))
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")

        self.indexes = self.calculate_indexes()

    def calculate_indexes(self):
        """This functions makes a list of every index of all the weights in the model. This is necessary to copy a
        percentage of the model weights instead of all the weights"""
        layers = self.model.layers
        indexes = []
        for layer in range(len(layers)):
            for node in range(layers[layer].weights[0].shape[0]):
                for weight in range(layers[layer].weights[0].shape[1]):
                    indexes.append((layer, node, weight))
        return indexes

    # def train(self, x, y):
    #     return self.model.fit(x, y)

    def save(self, location='models/model.h5'):
        """This function exports the current model"""
        self.model.save(location)

    def load_weights(self, new_weights):
        print(self.model.weights)

    def load_model(self, location):
        self.model = tf.keras.models.load_model(location)

    def __str__(self):
        return self.model.summary()
