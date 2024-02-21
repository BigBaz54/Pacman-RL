import random


class GameEnv:
    def __init__(self, grid):
        if len(grid) == 0 or len(grid[0]) == 0:
            raise ValueError("Grid should be non-empty")
        self.grid = grid
        self.num_rows = len(grid)
        self.num_cols = len(grid[0])
        # Pacman always starts in the bottom left corner of the grid
        self.state = (self.num_rows - 1) * self.num_cols


    def state_to_position(self, state):
        return state // self.num_cols, state % self.num_cols


    def position_to_state(self, position):
        return position[0] * self.num_cols + position[1]


    def get_reward(self, state):
        """
        Given the current state, return the reward for this state.
        :param state: current state (int)
        :return: reward (float)
        """
        n, m = self.state_to_position(state)
        if self.grid[n][m] == 0:
            return -0.04
        elif self.grid[n][m] == 1:
            return 1
        elif self.grid[n][m] == 2:
            return -1
        else:
            return float('nan')


    def get_next_state(self, state, action):
        """
        Given the current state and action, return the next state after taking the action.
        If the action leads to bumping into a wall, return the current state.
        :param state: current state (int)
        :param action: action to be taken ('up', 'down', 'left', 'right')
        :return: next state (int)
        """
        n, m = self.state_to_position(state)
        # Walls are represented by 3 in the grid
        if action == 'up' and n > 0 and self.grid[n - 1][m] != 3:
            return self.position_to_state((n - 1, m))
        elif action == 'down' and n < self.num_rows - 1 and self.grid[n + 1][m] != 3:
            return self.position_to_state((n + 1, m))
        elif action == 'left' and m > 0 and self.grid[n][m - 1] != 3:
            return self.position_to_state((n, m - 1))
        elif action == 'right' and m < self.num_cols - 1 and self.grid[n][m + 1] != 3:
            return self.position_to_state((n, m + 1))
        else:
            return state
    

    def get_possible_next_states(self, state, action):
        """
        Given the current state and action, return the list of possible next states after taking the action with their probabilities.
        If the action leads to bumping into a wall, it results in staying in the current state with the same probability.
        :param state: current state (int)
        :param action: action to be taken ('up', 'down', 'left', 'right')
        :return: list of possible next states with their probabilities (list of tuples (next_state, probability))
        """
        states = [(state, 0.0)]

        if action == 'up' or action == 'down':
            next_state = self.get_next_state(state, action)
            if next_state == state:
                states[0] = (state, states[0][1] + 0.8)
            else:
                states.append((next_state, 0.8))
            for a in ['left', 'right']:
                next_state = self.get_next_state(state, a)
                if next_state == state:
                    states[0] = (state, states[0][1] + 0.1)
                else:
                    states.append((next_state, 0.1))
        elif action == 'left' or action == 'right':
            next_state = self.get_next_state(state, action)
            if next_state == state:
                states[0] = (state, states[0][1] + 0.8)
            else:
                states.append((next_state, 0.8))
            for a in ['up', 'down']:
                next_state = self.get_next_state(state, a)
                if next_state == state:
                    states[0] = (state, states[0][1] + 0.1)
                else:
                    states.append((next_state, 0.1))

        if states[0][1] < 1e-12:
            states = states[1:]
        
        return states


    def draw_next_state(self, state, possible_next_states):
        """
        Given the possible next states and their probabilities, draw the next state according to the probabilities.
        :param state: current state (int)
        :param possible_next_states: list of possible next states with their probabilities (list of tuples (next_state, probability))
        :return: next state (int)
        """
        r = random.random()
        cum = 0
        for next_state, probability in possible_next_states:
            cum += probability
            if r < cum:
                return next_state
        return state


    def print_grid(self):
        """
        Print the grid to the console.
        """
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (i, j) == self.state_to_position(self.state):
                    print('P', end=' ')
                elif self.grid[i][j] == 0:
                    print('.', end=' ')
                elif self.grid[i][j] == 1:
                    print('R', end=' ')
                elif self.grid[i][j] == 2:
                    print('G', end=' ')
                else:
                    print('#', end=' ')
            print()
        print()


    def test_env(self):
        """
        Launches the game in the console.
        The grid is printed and the user is asked to input the next action.
        Press z, q, s, or d to move the Pacman.
        Pressing 'c' will quit the game, pressing 'r' will reset the game, 
        and pressing 'a' will display the available actions and possible next states.
        """
        self.print_grid()
        while True:
            action = input("Next action: ")
            if action == 'z':
                possible_next_states = self.get_possible_next_states(self.state, 'up')
                self.state = self.draw_next_state(self.state, possible_next_states)
            elif action == 's':
                possible_next_states = self.get_possible_next_states(self.state, 'down')
                self.state = self.draw_next_state(self.state, possible_next_states)
            elif action == 'q':
                possible_next_states = self.get_possible_next_states(self.state, 'left')
                self.state = self.draw_next_state(self.state, possible_next_states)
            elif action == 'd':
                possible_next_states = self.get_possible_next_states(self.state, 'right')
                self.state = self.draw_next_state(self.state, possible_next_states)
            elif action == 'c':
                break
            elif action == 'r':
                self.state = (self.num_rows - 1) * self.num_cols
            elif action == 'a':
                print("Possible actions:")
                for a in ['up', 'down', 'left', 'right']:
                    print(a, end=' ')
                    print(":", self.get_possible_next_states(self.state, a))
                print()
            else:
                print("Invalid action")
            self.print_grid()


if __name__ == "__main__":
    grid = [[0, 0, 0, 1],
            [0, 3, 0, 2],
            [0, 0, 0, 0]]
    env = GameEnv(grid)
    env.test_env()
