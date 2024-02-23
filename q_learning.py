from game_env import GameEnv
import random


class QLearning():
    def __init__(self, path_to_settings, epsilon=1, eps_decay=0.99):
        self.game_env, self.gamma, self.alpha, self.nb_episodes = self.parse_settings_file(path_to_settings)
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.q_values = {state: {'up': 0, 'down': 0, 'left': 0, 'right': 0} for state in range(self.game_env.num_cols * self.game_env.num_rows)}
        self.policy = {state: '' for state in range(self.game_env.num_cols * self.game_env.num_rows)}


    def parse_settings_file(self, path_to_settings):
        """
        Parses the settings file and create a GameEnv object.
        :param path_to_settings: path to the settings file (str)
        :return: GameEnv object, gamma, alpha, nb_episodes
        """
        with open(path_to_settings, 'r') as f:
            settings = f.readlines()
        settings = [x.strip() for x in settings]
        nb_episodes = int(settings[-1])
        alpha = float(settings[-2])
        gamma = float(settings[-3])
        grid = [[int(c) for c in list(x)] for x in settings[:-3]]
        env = GameEnv(grid)

        return env, gamma, alpha, nb_episodes


    def get_next_action(self, state):
        """
        Get the next action to take and update the epsilon value.
        :param state: current state (int)
        :return: next action (str)
        """
        self.epsilon = self.epsilon * self.eps_decay
        if random.uniform(0, 1) < self.epsilon or self.policy[state] == '':
            return random.choice(['up', 'down', 'left', 'right'])
        else:
            return self.policy[state]


    def update_q_values(self, prev_state, action, reward, curr_state):
        """
        Update the Q-values.
        :param prev_state: previous state (int)
        :param action: action taken (str)
        :param reward: reward received (int)
        :param curr_state: current state (int)
        """
        self.q_values[prev_state][action] = self.q_values[prev_state][action] + self.alpha * (reward + self.gamma * max(self.q_values[curr_state].values()) - self.q_values[prev_state][action])


    def update_policy(self, state):
        """
        Update the policy.
        :param state: current state (int)
        """
        self.policy[state] = max(self.q_values[state], key=lambda x: self.q_values[state][x])


    def train(self):
        """
        Train the agent.
        """
        for episode in range(self.nb_episodes):
            self.game_env.reset()
            curr_state = self.game_env.state
            terminal = False
            while not terminal:
                terminal = self.game_env.is_terminal(curr_state)
                if episode == 0:
                    action = random.choice(['up', 'down', 'left', 'right'])
                else:
                    action = self.get_next_action(curr_state)
                prev_state = curr_state
                possible_next_states = self.game_env.get_possible_next_states(curr_state, action)
                curr_state = self.game_env.draw_next_state(prev_state, possible_next_states)
                reward = self.game_env.get_reward(curr_state)
                self.update_q_values(prev_state, action, reward, curr_state)
                self.update_policy(curr_state)


    def print_policy(self):
            """
            Print the policy to the console.
            """
            print('\n' + '=' * 100 + '\n')
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


    def print_q_values(self):
        """
        Print the Q-values to the console.
        """
        print('\n' + '=' * 100 + '\n')
        print('Q-values\n')
        for i in range(self.game_env.num_rows):
            for actions in [[('up', '↑'), ('down', '↓')], [('left', '←'), ('right', '→')]]:
                for j in range(self.game_env.num_cols):
                    if self.game_env.grid[i][j] == 3:
                        print('#'*17, end=' ')
                        if j < self.game_env.num_cols - 1:
                            print('|', end=' ')
                    else:
                        for action in actions:
                            print(f'{action[1]}:{self.q_values[self.game_env.position_to_state((i, j))][action[0]]:^6.2f}', end=' ')
                        if j < self.game_env.num_cols - 1:
                            print('|', end=' ')
                print()
            if i < self.game_env.num_rows - 1:
                print('-' * 19 * self.game_env.num_cols)


if __name__ == "__main__":
    q_learning = QLearning('Q-Learning.txt')
    q_learning.train()
    q_learning.print_q_values()
    q_learning.print_policy()