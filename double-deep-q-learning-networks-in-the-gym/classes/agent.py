from classes.memory import Memory
from classes.functionapproximator import FunctionApproximator
import numpy as np


class Agent:
    def __init__(self, memory_size, gamma, alpha, epsilon) -> None:
        self.gamma = gamma     
        self.alpha = alpha
        self.epsilon = epsilon

        self.memory = Memory(memory_size)
        self.policy_network = FunctionApproximator()
        self.target_network = FunctionApproximator()

    def learn(self, sample_size):
        sample_size = min(len(self.memory.transitions), sample_size)
        batch = self.memory.sample(sample_size)

        targets = []
        next_states = []
        for transition in batch:
            next_state = transition.next_state
            next_states.append(next_state)
            best_action = np.argmax(self.policy_network.model.predict(np.array([next_state]))[0])
            if not transition.done:
                q_value_best_action = self.target_network.model.predict(np.array([next_state]))[0][best_action]
            else:
                q_value_best_action = 0
            target = transition.reward + self.gamma * q_value_best_action
            targets.append(np.array([target]))
        
        targets = np.array(targets)
        next_states = np.array(next_states)
        # print(targets.shape, next_states.shape)
        self.policy_network.train(next_states, targets)
        

