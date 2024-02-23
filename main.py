from value_iteration import ValueIteration
from q_learning import QLearning


# Create an instance of the ValueIteration class that reads the environment from the file 'value-iteration.txt'
# and create its own GameEnv object.
vi = ValueIteration('value-iteration.txt')
# Run the value iteration algorithm and store the values for each state in self.values.
vi.value_iteration(trace=True)
# Compute the final policy and store the policy for each state in self.policy.
vi.compute_policy(trace=True)


#Create an instance of the QLearning class that reads the environment from the file 'Q-Learning.txt'
# and create its own GameEnv object.
ql = QLearning('Q-Learning.txt')
# Run the Q-learning algorithm and store the Q-values for each state-action pair in self.q_values
# and the policy for each state in self.policy.
ql.train(trace=True)
