from agent import Agent 
from maze import Maze
from policy import Policy
import unittest   # The test framework


class Test_Agent(unittest.TestCase):
    def setUp(self) -> None:
        self.maze = Maze(
                    lenght=4,
                    height=4,
                    all_rewards=-1,
                    special_rewards={
                        (3, 0): 40,
                        (2, 1): -10,
                        (3, 1): -10,
                        (0, 3): 10,
                        (1, 3): -2},
                    end_positions=[(3, 0), (0, 3)]
                )
        self.policy = Policy(lenght=4, height=4)
        self.agent = Agent(self.maze, self.policy, (2, 3), 1)

    def generate_empty_value_function(self):
        expected_output = [[0]]
        self.assertEqual(self.agent.generate_empty_value_function(1, 1), expected_output)

    # def test_decrement(self):
    #     self.assertEqual(inc_dec.decrement(3), 4)

if __name__ == '__main__':
    unittest.main()
