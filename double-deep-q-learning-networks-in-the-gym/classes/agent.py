from classes.memory import Memory
from classes.functionapproximator import FunctionApproximator
from classes.epsilongreedypolicy import EpsilonGreedyPolicy
from random import sample
import numpy as np
import time


class Agent:
    def __init__(self, n_actions, memory_size, gamma, alpha, epsilon, min_epsilon, epsilon_decay) -> None:
        self.gamma = gamma     
        self.alpha = alpha
        self.epsilon = epsilon

        self.memory = Memory(memory_size)
        self.policy_network = FunctionApproximator()
        self.target_network = FunctionApproximator()
        self.policy = EpsilonGreedyPolicy(n_actions, epsilon, min_epsilon, epsilon_decay)

    def learn(self, sample_size):
        sample_size = min(len(self.memory.transitions), sample_size)
        batch = self.memory.sample(sample_size)

        # take a mini-batch from replay experience
        cur_batch_size = min(len(self.memory.transitions), sample_size)
        mini_batch = sample(self.memory.transitions, cur_batch_size)

        # batch data
        sample_states = np.ndarray(shape=(cur_batch_size, 8))  # replace 128 with cur_batch_size
        sample_actions = np.ndarray(shape=(cur_batch_size, 1))
        sample_rewards = np.ndarray(shape=(cur_batch_size, 1))
        sample_next_states = np.ndarray(shape=(cur_batch_size, 8))
        sample_dones = np.ndarray(shape=(cur_batch_size, 1))

        temp = 0
        for exp in mini_batch:
            sample_states[temp] = exp.state
            sample_actions[temp] = exp.action
            sample_rewards[temp] = exp.reward
            sample_next_states[temp] = exp.next_state
            sample_dones[temp] = exp.done
            temp += 1

        sample_qhat_next = self.target_network.model.predict(sample_next_states)

        # set all Q values terminal states to 0
        sample_qhat_next = sample_qhat_next * (np.ones(shape=sample_dones.shape) - sample_dones)
        # choose max action for each state
        sample_qhat_next = np.max(sample_qhat_next, axis=1)

        sample_qhat = self.policy_network.model.predict(sample_states)

        for i in range(cur_batch_size):
            a = sample_actions[i, 0]
            sample_qhat[i, int(a)] = sample_rewards[i] + self.gamma * sample_qhat_next[i]

        q_target = sample_qhat

        self.policy_network.model.fit(sample_states, q_target, epochs=1, verbose=0)

    def copy_model(self, tau=1):
        if tau >= 1:
            self.target_network.model.set_weights(self.policy_network.model.get_weights()) 
        amount_of_weights_to_change = int(len(self.policy_network.indexes) * tau)
        indexes_to_change = sample(self.policy_network.indexes, amount_of_weights_to_change)
        for index_to_change in indexes_to_change:
            i_layer, i_node, i_weight = index_to_change
            new_weight = self.policy_network.model.layers[i_layer].weights[0][i_node][i_weight]
            weights = self.target_network.model.layers[i_layer].get_weights()
            weights[0][i_node, i_weight] = new_weight.numpy()
            self.target_network.model.layers[i_layer].set_weights(weights)

    def save_models(self):
        self.policy_network.save(f'double-deep-q-learning-networks-in-the-gym/models/policy-model-{time.strftime("%Y%m%d-%H%M%S")}')
        self.target_network.save(f'double-deep-q-learning-networks-in-the-gym/models/target-model-{time.strftime("%Y%m%d-%H%M%S")}')
