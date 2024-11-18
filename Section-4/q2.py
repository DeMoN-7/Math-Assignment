# 2. Write a Python program to simulate an HMM with given transition and 
# emission matrices and generate a sequence of observations

import numpy as np

def simulate_hmm(start_prob, trans_prob, emit_prob, num_steps):
    n_states = len(start_prob)
    states = np.zeros(num_steps, dtype=int)
    observations = np.zeros(num_steps, dtype=int)

    states[0] = np.random.choice(n_states, p=start_prob)
    observations[0] = np.random.choice(len(emit_prob[0]), p=emit_prob[states[0]] / np.sum(emit_prob[states[0]]))

    for t in range(1, num_steps):
        states[t] = np.random.choice(n_states, p=trans_prob[states[t-1]])
        observations[t] = np.random.choice(len(emit_prob[0]), p=emit_prob[states[t]] / np.sum(emit_prob[states[t]]))

    return states, observations

start_prob = np.array([0.6, 0.4])
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
emit_prob = np.array([[0.1, 0.4], [0.6, 0.3]])

num_steps = 10
states, observations = simulate_hmm(start_prob, trans_prob, emit_prob, num_steps)

print("Generated States:", states)
print("Generated Observations:", observations)

