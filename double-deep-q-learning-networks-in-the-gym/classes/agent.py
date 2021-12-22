from classes.memory import Memory
from classes.functionapproximator import FunctionApproximator
from classes.epsilongreedypolicy import EpsilonGreedyPolicy
from random import sample
import numpy as np
import time


"""
        This function learns the policy network
        The algorithm is inspired by: https://github.com/anh-nn01/Lunar-Lander-Double-Deep-Q-Networks
        """


class Agent:
    """The agent class (lunar lander)"""
    def __init__(self, n_actions: int, n_states: int, memory_size: int, gamma: float, alpha: float, epsilon: float, min_epsilon: float, epsilon_decay: float) -> None:
        """
        Constructor for the Agent class

        Args:
            n_actions (int): The amount of actions that can be taken.
            n_states (int): The lenght of the state element.
            memory_size (int): The size of the memory.
            gamma (float): The gamm / discount.
            alpha (float): The alpha / learning rate for the Adam optimizer.
            epsilon (float): The epsilon to use.
            min_epsilon (float): The lowest the epsilon can go.
            epsilon_decay (float): The factor the epsilon decays by.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_states = n_states

        self.memory = Memory(memory_size)
        self.policy_network = FunctionApproximator(n_states, n_actions, alpha)
        self.target_network = FunctionApproximator(n_states, n_actions, alpha)
        self.copy_model()
        self.policy = EpsilonGreedyPolicy(n_actions, epsilon, min_epsilon, epsilon_decay, n_states)

    def learn(self, sample_size: int) -> None:
        """
        Learns the agent.
        This algorithm is inspired by: https://github.com/anh-nn01/Lunar-Lander-Double-Deep-Q-Networks

        Args:
            sample_size (int): The size of sample to take from the memory.
        """
        # Create batch from memory
        sample_size = min(len(self.memory.transitions), sample_size)
        batch = self.memory.sample(sample_size)

        # Transform the batch of transitions to numpy arrays
        sample_states = np.ndarray(shape=(sample_size, self.n_states))
        sample_actions = np.ndarray(shape=(sample_size, 1))
        sample_rewards = np.ndarray(shape=(sample_size, 1))
        sample_next_states = np.ndarray(shape=(sample_size, self.n_states))
        sample_dones = np.ndarray(shape=(sample_size, 1))

        for index_transition, transition in enumerate(batch):
            sample_states[index_transition] = transition.state
            sample_actions[index_transition] = transition.action
            sample_rewards[index_transition] = transition.reward
            sample_next_states[index_transition] = transition.next_state
            sample_dones[index_transition] = transition.done

        policy_predictions = self.policy_network.model.predict(sample_states)

        target_next_states_predictions = self.target_network.model.predict(sample_next_states)

        # set all Q values terminal states to 0
        target_next_states_predictions = target_next_states_predictions * (
                    np.ones(shape=sample_dones.shape) - sample_dones)
        # choose max action for each state
        target_next_q = np.max(target_next_states_predictions, axis=1)

        for i in range(sample_size):
            a = sample_actions[i, 0]
            policy_predictions[i, int(a)] = sample_rewards[i] + self.gamma * target_next_q[i]

        target = policy_predictions
        self.policy_network.model.fit(sample_states, target, epochs=1, verbose=0)

    def learn_david(self, sample_size: int) -> None:
        """
        Learns the agent, or at least it is supposed to.
        This code is based on Canvas.

        Args:
            sample_size (int): The size of sample to take from the memory.
        """
        sample_size = min(len(self.memory.transitions), sample_size)
        batch = self.memory.sample(sample_size)

        sample_states = np.ndarray(shape=(sample_size, self.n_states))
        sample_actions = np.ndarray(shape=(sample_size, 1))
        sample_rewards = np.ndarray(shape=(sample_size, 1))
        sample_next_states = np.ndarray(shape=(sample_size, self.n_states))
        sample_dones = np.ndarray(shape=(sample_size, 1))

        for index_transition, transition in enumerate(batch):
            sample_states[index_transition] = transition.state
            sample_actions[index_transition] = transition.action
            sample_rewards[index_transition] = transition.reward
            sample_next_states[index_transition] = transition.next_state
            sample_dones[index_transition] = transition.done

        q_values_states_policy = self.policy_network.model.predict(sample_next_states)
        q_values_next_states_policy = self.policy_network.model.predict(sample_next_states)
        best_actions_next_states_policy = np.argmax(q_values_next_states_policy, axis=1)

        q_values_next_states_target = self.target_network.model.predict(sample_next_states)
        q_values_next_states_target = q_values_next_states_target * (np.ones(shape=sample_dones.shape) - sample_dones)

        for i in range(sample_size):
            best_action = best_actions_next_states_policy[i]
            q_value_next_state_target = q_values_next_states_target[i][best_action]
            target = sample_rewards[i][0] + self.gamma * q_value_next_state_target
            taken_action = int(sample_actions[i][0])
            q_values_states_policy[i][taken_action] = target

        q_target = q_values_next_states_policy
        self.policy_network.model.fit(sample_states, q_target, epochs=1, verbose=0)

    def copy_model(self, tau: int = 1) -> None:
        """
        copies the weights from the policy network to the target network

        Args:
            tau (int, optional): The percentage of the model to copy. Defaults to 1.
        """
        if tau >= 1:  # If tau = 1 all the weights will be copied
            self.target_network.model.set_weights(self.policy_network.model.get_weights())
        else:  # Else a percentage of the weights will be copied
            amount_of_weights_to_change = int(len(self.policy_network.indexes) * tau)
            indexes_to_change = sample(self.policy_network.indexes, amount_of_weights_to_change)
            for index_to_change in indexes_to_change:
                i_layer, i_node, i_weight = index_to_change
                new_weight = self.policy_network.model.layers[i_layer].weights[0][i_node][i_weight]
                weights = self.target_network.model.layers[i_layer].get_weights()
                weights[0][i_node, i_weight] = new_weight.numpy()
                self.target_network.model.layers[i_layer].set_weights(weights)

    def save_models(self):
        """
        This function exports the policy and the target network
        """
        self.policy_network.save(
            f'double-deep-q-learning-networks-in-the-gym/models/policy-model-{time.strftime("%Y%m%d-%H%M%S")}')
        self.target_network.save(
            f'double-deep-q-learning-networks-in-the-gym/models/target-model-{time.strftime("%Y%m%d-%H%M%S")}')
