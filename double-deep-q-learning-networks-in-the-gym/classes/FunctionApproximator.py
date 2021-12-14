from keras import layers, Sequential, optimizers


class FunctionApproximator:
    def __init__(self):
        self.model = Sequential()
        self.model.add(layers.Dense(8, input_shape=(8,)))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(32, activation="relu"))
        self.model.add(layers.Dense(1, activation="sigmoid"))
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="rms")


    def __str__(self):
        return "...."
