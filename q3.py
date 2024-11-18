# Write a Python program to compute the expected time until absorption in a 
# birth-death process for a given set of states

import numpy as np

def expected_absorption_time(birth_rate, death_rate, num_states):
    Q = np.zeros((num_states, num_states))
    for i in range(num_states - 1):
        Q[i, i + 1] = birth_rate
        Q[i + 1, i] = death_rate
        Q[i, i] = -(birth_rate + death_rate)
    Q[-1, -1] = 0
    Q = Q[:-1, :-1]
    N = -np.linalg.inv(Q)
    absorption_times = N.sum(axis=1)
    return absorption_times

birth_rate = 0.1
death_rate = 0.05
num_states = 10

absorption_times = expected_absorption_time(birth_rate, death_rate, num_states)
print("Expected Absorption Times:", absorption_times)
