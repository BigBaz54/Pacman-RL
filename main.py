from value_iteration import ValueIteration


# Create an instance of the ValueIteration class that reads the environment from the file 'value-iteration.txt'
# and create its own GameEnv object.
vi = ValueIteration('value-iteration.txt')
# Run the value iteration algorithm and store the values for each state in self.values.
vi.value_iteration(trace=True)
# Compute the final policy and store the policy for each state in self.policy.
vi.compute_policy(trace=True)
