import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class FunctionApproximator:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(8,), activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(4, activation="linear"))
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")

        self.indexes = self.calculate_indexes()

    def calculate_indexes(self):
        layers = self.model.layers
        indexes = []
        for layer in range(len(layers)):
            for node in range(layers[layer].weights[0].shape[0]):
                for weight in range(layers[layer].weights[0].shape[1]):
                    indexes.append((layer, node, weight))
        return indexes

    def train(self, x, y):
        return self.model.fit(x, y)

    def save(self, location='models/model.h5'):
        self.model.save(location)

    def load_weights(self, new_weights):
        print(self.model.weights)

    def load_model(self, location):
        self.model = tf.keras.models.load_model(location)

    def __str__(self):
        return "...."
