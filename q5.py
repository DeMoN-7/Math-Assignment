# . Write a Python program to calculate the probability that a birth-death process reaches a 
# specific state after f time units, starting from an initial state.

import numpy as np

def state_probability(birth_rate, death_rate, initial_state, target_state, time_units, max_state):
    P = np.zeros((max_state + 1, max_state + 1))
    
    for i in range(max_state):
        P[i, i + 1] = birth_rate
        P[i + 1, i] = death_rate
        P[i, i] = -(birth_rate + death_rate)
    
    P[-1, -1] = 0
    P = P[:-1, :-1]
    
    transition_matrix = np.linalg.matrix_power(P, time_units)
    
    probability = transition_matrix[initial_state, target_state]
    return probability

birth_rate = 0.1
death_rate = 0.05
initial_state = 2
target_state = 5
time_units = 10
max_state = 10

probability = state_probability(birth_rate, death_rate, initial_state, target_state, time_units, max_state)
print(f"Probability of reaching state {target_state} from state {initial_state} after {time_units} time units: {probability}")
