import tensorflow as tf 
from keras import layers, Sequential, optimizers


class FunctionApproximator:
    def __init__(self):
        self.model = Sequential()
        self.model.add(layers.Dense(8, input_shape=(8,)))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(4, activation="linear"))
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())

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

    def save(self, location='models/modelv1.h5'):
        self.model.save(location)

    def load_weights(self, new_weights):
        print(self.model.weights)

    def __str__(self):
        return "...."
