from game_env import GameEnv


class ValueIteration:
    def __init__(self, path_to_settings):
        self.game_env, self.gamma, self.epsilon = self.parse_settings_file(path_to_settings)
        self.values = [0 for _ in range(self.game_env.num_cols * self.game_env.num_rows)]
        self.policy = ['' for _ in range(self.game_env.num_cols * self.game_env.num_rows)]


    def parse_settings_file(self, path_to_settings):
        """
        Parses the settings file and create a GameEnv object.
        :param path_to_settings: path to the settings file (str)
        :return: GameEnv object, gamma, epsilon
        """
        with open(path_to_settings, 'r') as f:
            settings = f.readlines()
        settings = [x.strip() for x in settings]
        epsilon = float(settings[-1])
        gamma = float(settings[-2])
        grid = [[int(c) for c in list(x)] for x in settings[:-2]]
        env = GameEnv(grid)

        return env, gamma, epsilon


    def value_iteration(self, trace=False, verbose=False):
        """
        Runs the value iteration algorithm and stores the values for each state in self.values.
        :param trace: (bool) if True, write the values to the file 'log-file_VI.txt' after each iteration
        :param verbose: (bool) if True, print the values to the console after each iteration
        """
        c = 0
        delta = 0
        while True:
            c += 1
            delta = 0

            if trace:
                f = open('log-file_VI.txt', 'a', encoding='utf-8')
                f.write('\n' + '=' * 50 + '\n\n')
                f.write(f'Iteration {c}\n\n')
            if verbose:
                self.print_values(c, delta)
            
            old_values = self.values.copy()
            for state in range(self.game_env.num_cols * self.game_env.num_rows):

                if trace:
                    f.write(f'U[{state}] = max(')
                
                max_v = float('-inf')
                for action in ['up', 'down', 'left', 'right']:
                    next_states = self.game_env.get_possible_next_states(state, action)
                    action_v = 0
                    for next_state, probability in next_states:

                        if trace:
                            f.write(f'{probability} * [{self.game_env.get_reward(next_state)} + {self.gamma} * {old_values[next_state]}]')
                            if next_state != next_states[-1][0]:
                                f.write(' + ')
                        
                        action_v += probability * (self.game_env.get_reward(next_state) + self.gamma * old_values[next_state])
                    
                    if trace:
                        if action != 'right':
                            f.write(', ')

                    max_v = max(max_v, action_v)
                self.values[state] = max_v

                if trace:
                    f.write(')\n')
                    f.write(f'     = {max_v:.5f}\n')

                delta += abs(old_values[state] - self.values[state])
            
            if trace:
                f.write('\nUpdated values:\n\n')
                for i in range(self.game_env.num_rows):
                    for j in range(self.game_env.num_cols):
                        if self.game_env.grid[i][j] == 3:
                            f.write('#' * 6)
                        else:
                            f.write(f'{self.values[i * self.game_env.num_cols + j]:^6.2f}')
                        if j < self.game_env.num_cols - 1:
                            f.write('|')
                    if i < self.game_env.num_rows - 1:
                        f.write('\n' + '-' * 7 * self.game_env.num_cols + '\n')
                f.write(f'\n\ndelta: {delta:.8f}\n')
            
            if delta < self.epsilon:
                break
        
        if trace:
            f.write('\ndelta < epsilon, algorithm converged\n')
            f.close()


    def compute_policy(self, trace=False, verbose=False):
        """
        Computes the policy for each state according to the computed values and stores it in self.policy.
        :param trace: (bool) if True, write the policy to the file 'log-file_VI.txt'
        :param verbose: (bool) if True, print the policy to the console
        """
        for state in range(self.game_env.num_cols * self.game_env.num_rows):
            max_v = float('-inf')
            for action in ['up', 'down', 'left', 'right']:
                next_states = self.game_env.get_possible_next_states(state, action)
                action_v = 0
                for next_state, probability in next_states:
                    action_v += probability * (self.game_env.get_reward(next_state) + self.gamma * self.values[next_state])
                if action_v > max_v:
                    max_v = action_v
                    self.policy[state] = action
        if trace:
            self.trace_policy()
        if verbose:
            self.print_policy()


    def print_policy(self):
        """
        Print the policy to the console.
        """
        print('\n' + '=' * 50 + '\n')
        print('Optimal policy\n')
        for i in range(self.game_env.num_rows):
            for j in range(self.game_env.num_cols):
                if self.game_env.grid[i][j] == 3:
                    print('#', end=' ')
                elif self.policy[self.game_env.position_to_state((i, j))] == 'up':
                    print('↑', end=' ')
                elif self.policy[self.game_env.position_to_state((i, j))] == 'down':
                    print('↓', end=' ')
                elif self.policy[self.game_env.position_to_state((i, j))] == 'left':
                    print('←', end=' ')
                elif self.policy[self.game_env.position_to_state((i, j))] == 'right':
                    print('→', end=' ')
                else:
                    raise ValueError('Invalid action')
            print()
        print()


    def print_values(self, c, delta):
        """
        Print the values to the console.
        :param c: (int) iteration number
        :param delta: (float) difference between the old and new values
        """
        print('\n' + '=' * 50 + '\n')
        print(f'Iteration {c}\n')
        for i in range(self.game_env.num_rows):
            for j in range(self.game_env.num_cols):
                if self.game_env.grid[i][j] == 3:
                    print('#' * 6, end='')
                else:
                    print(f"{self.values[i * self.game_env.num_cols + j]:^6.2f}", end='')
                if j < self.game_env.num_cols - 1:
                    print('|', end='')
            if i < self.game_env.num_rows - 1:
                print('\n' + '-' * 7 * self.game_env.num_cols)
        print(f'\n\ndelta: {delta:.8f}')


    def trace_policy(self):
        """
        Write the policy to the file 'log-file_VI.txt'.
        """
        with open('log-file_VI.txt', 'a', encoding='utf-8') as f:
            f.write('\n' + '=' * 50 + '\n\n')
            f.write('Optimal policy\n\n')
            for i in range(self.game_env.num_rows):
                for j in range(self.game_env.num_cols):
                    if self.game_env.grid[i][j] == 3:
                        f.write('# ')
                    elif self.policy[self.game_env.position_to_state((i, j))] == 'up':
                        f.write('↑ ')
                    elif self.policy[self.game_env.position_to_state((i, j))] == 'down':
                        f.write('↓ ')
                    elif self.policy[self.game_env.position_to_state((i, j))] == 'left':
                        f.write('← ')
                    elif self.policy[self.game_env.position_to_state((i, j))] == 'right':
                        f.write('→ ')
                    else:
                        raise ValueError('Invalid action')
                f.write('\n')


if __name__ == '__main__':
    vi = ValueIteration('value-iteration.txt')
    vi.value_iteration(verbose=True, trace=True)
    vi.compute_policy(verbose=True, trace=True)
