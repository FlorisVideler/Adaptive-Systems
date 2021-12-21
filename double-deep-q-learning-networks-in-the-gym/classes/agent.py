from classes.memory import Memory
from classes.functionapproximator import FunctionApproximator
from classes.epsilongreedypolicy import EpsilonGreedyPolicy
from random import sample
import numpy as np
import time


class Agent:
    """The agent class (lunar lander)"""
    def __init__(self, n_actions, n_states, memory_size, gamma, alpha, epsilon, min_epsilon, epsilon_decay) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_states = n_states

        self.memory = Memory(memory_size)
        self.policy_network = FunctionApproximator(n_states, n_actions)
        self.target_network = FunctionApproximator(n_states, n_actions)
        self.copy_model()
        self.policy = EpsilonGreedyPolicy(n_actions, epsilon, min_epsilon, epsilon_decay, n_states)

    def learn(self, sample_size):
        """
        This function learns the policy network
        The algorithm is inspired by: https://github.com/anh-nn01/Lunar-Lander-Double-Deep-Q-Networks
        """
        # Create batch from memory
        sample_size = min(len(self.memory.transitions), sample_size)
        batch = self.memory.sample(sample_size)

        # Onze oplossing + numpy

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

        # for index_transition, transition in enumerate(batch):
        #     sample_states[index_transition] = transition[0]
        #     sample_actions[index_transition] = transition[1]
        #     sample_rewards[index_transition] = transition[2]
        #     sample_next_states[index_transition] = transition[3]
        #     sample_dones[index_transition] = transition[4]

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

        # policy_next_states_predictions = self.policy_network.model.predict(sample_next_states)

        # Wat is de beste actie?

        # policy_next_states_best_actions = np.argmax(policy_next_states_predictions, axis=1)
        # f = lambda a: [a]
        # B
        # policy_next_states_best_actions = np.array(list(map(f, policy_next_states_best_actions)))

        # target_next_states_predictions = self.target_network.model.predict(sample_next_states)

        # Na de indexing? 
        # target_next_states_predictions = target_next_states_predictions * (np.ones(shape = sample_dones.shape) - sample_dones)

        # m,n = target_next_states_predictions.shape
        # q_value_next_state_action = np.take(target_next_states_predictions, policy_next_states_best_actions + np.arange(m)[:,None])

        # q_value_next_state_action = np.ndarray(shape = (sample_size, 1))

        # for index_item, item in np.ndenumerate(policy_next_states_best_actions):
        #     q_value_next_state_action[index_item] = target_next_states_predictions[index_item, item]

        # for i in range(sample_size):
        #     a = sample_actions[i,0]
        #     reward = sample_rewards[i, 0]
        #     next_state_best_action_policy = policy_next_states_best_actions[i]
        #     q_value_best_step_target = target_next_states_predictions[i, next_state_best_action_policy]
        #     policy_next_states_predictions[i,int(a)] = reward + self.gamma * q_value_best_step_target

        # target = policy_next_states_predictions
        # self.policy_network.model.fit(sample_states, target, epochs=1, verbose=0)

        # targets = []
        # states = []
        # for transition in batch:
        #     next_state = transition.next_state
        #     state = transition.state
        #     action = transition.action
        #     states.append(state)
        #     best_action = np.argmax(self.policy_network.model.predict(np.array([next_state]))[0])
        #     if not transition.done:
        #         q_value_best_action = self.target_network.model.predict(np.array([next_state]))[0][best_action]
        #         # Moet target 0 zijn als next state done is
        #     else:
        #         q_value_best_action = 0

        #     target = transition.reward + self.gamma * q_value_best_action

        #     current_predictions = self.policy_network.model.predict(np.array([state]))[0]
        #     current_predictions[action] = target
        #     targets.append(np.array([current_predictions]))

        # targets = np.array(targets)
        # states = np.array(states)
        # # print(targets.shape, next_states.shape)
        # #TODO: Next state of state?
        # self.policy_network.train(states, targets)
        # print(f'Epsilon: {self.policy.epsilon}')

    def learn_david(self, sample_size):
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
        # States of next_states?
        self.policy_network.model.fit(sample_states, q_target, epochs=1, verbose=0)

    def copy_model(self, tau=1):
        """This function copies the weights from the policy network to the target network"""
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
        """This function exports the policy and the target network"""
        self.policy_network.save(
            f'double-deep-q-learning-networks-in-the-gym/models/policy-model-{time.strftime("%Y%m%d-%H%M%S")}')
        self.target_network.save(
            f'double-deep-q-learning-networks-in-the-gym/models/target-model-{time.strftime("%Y%m%d-%H%M%S")}')
