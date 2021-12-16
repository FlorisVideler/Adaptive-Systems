from os import stat
from classes.memory import Memory
from classes.functionapproximator import FunctionApproximator
from classes.epsilongreedypolicy import EpsilonGreedyPolicy
import copy
from random import sample
import numpy as np
import tensorflow as tf 
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

        targets = []
        states = []
        for transition in batch:
            next_state = transition.next_state
            state = transition.state
            action = transition.action
            states.append(next_state)
            best_action = np.argmax(self.policy_network.model.predict(np.array([next_state]))[0])
            if not transition.done:
                q_value_best_action = self.target_network.model.predict(np.array([next_state]))[0][best_action]
            else:
                q_value_best_action = 0
            target = transition.reward + self.gamma * q_value_best_action
            
            current_predictions = self.policy_network.model.predict(np.array([state]))[0]
            current_predictions[action] = target
            targets.append(np.array([current_predictions]))

        targets = np.array(targets)
        states = np.array(states)
        # print(targets.shape, next_states.shape)
        #TODO: Next state of state?
        self.policy_network.train(states, targets)
        self.policy.decay_epsilon()

    def copy_model(self, tau=1):
        amount_of_weights_to_change = int(len(self.policy_network.indexes) * tau)
        indexes_to_change = sample(self.policy_network.indexes, amount_of_weights_to_change)
        for index_to_change in indexes_to_change:
            i_layer, i_node, i_weight = index_to_change
            new_weight = self.policy_network.model.layers[i_layer].weights[0][i_node][i_weight]
            weights = self.target_network.model.layers[i_layer].get_weights()
            weights[0][i_node, i_weight] = new_weight.numpy()
            self.target_network.model.layers[i_layer].set_weights(weights)

    def save_models(self):
        self.policy_network.save(f'../models/policy-model-{time.strftime("%Y%m%d-%H%M%S")}')
        self.target_network.save(f'../models/target-model-{time.strftime("%Y%m%d-%H%M%S")}')
