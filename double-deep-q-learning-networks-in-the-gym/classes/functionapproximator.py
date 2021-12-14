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

    def train(self, x, y):
        print(x.shape, y.shape)
        return self.model.fit(x, y)

    def save(self, location='models/modelv1.h5'):
        self.model.save(location)

    def __str__(self):
        return "...."
