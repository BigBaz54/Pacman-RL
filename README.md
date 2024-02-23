# Pacman-RL

## Input format

To run the code, you need to have a file called `value-iteration.txt` and a file called `Q-Learning.txt` in the same directory as the code.
These files should follow the format of the example files provided in the repository, i.e.:
    
- `value-iteration.txt` :
```
G R I D
G R I D
G R I D
G R I D
gamma
epsilon
```
- `Q-Learning.txt` :
```
G R I D G R I D
G R I D G R I D
G R I D G R I D
gamma
alpha
nb_episodes
```

## Running the code

To run the code, you need to run the following command in the terminal:
```
python3 main.py
```
This will run the code for both the value iteration and Q-learning algorithms, and will output the results in the files `log-file_VI.txt` and `log-file_QL.txt` respectively.