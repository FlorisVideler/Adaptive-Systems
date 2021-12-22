from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class FunctionApproximator:
    def __init__(self, n_states: int, n_actions: int, alpha: float) -> None:
        """
        The constructor of the function approximator class. Builds the model.

        Args:
            n_states (int): The lenght of the state object.
            n_actions (int): The amount of actions that are possible
            alpha (float): The alpha / learning rate for the Adam optimizer.
        """
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=n_states, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(n_actions, activation="linear"))
        self.model.compile(optimizer=Adam(learning_rate=alpha), loss="mse")

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

    def __str__(self):
        return self.model.summary()
