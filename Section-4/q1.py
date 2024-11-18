# . Implement the Viterbi algorithm in Python to find the most likely sequence 
# of hidden states given an observed sequence and an HMM.

import numpy as np

def viterbi(observations, states, start_prob, trans_prob, emit_prob):
    n_states = len(states)
    n_obs = len(observations)
    V = np.zeros((n_states, n_obs))
    backpointer = np.zeros((n_states, n_obs), dtype=int)

    for state in range(n_states):
        V[state, 0] = start_prob[state] * emit_prob[state, observations[0]]
        backpointer[state, 0] = 0

    for t in range(1, n_obs):
        for state in range(n_states):
            max_prob = -1
            best_prev_state = 0
            for prev_state in range(n_states):
                prob = V[prev_state, t - 1] * trans_prob[prev_state, state] * emit_prob[state, observations[t]]
                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = prev_state
            V[state, t] = max_prob
            backpointer[state, t] = best_prev_state

    best_path = np.zeros(n_obs, dtype=int)
    best_path[-1] = np.argmax(V[:, -1])
    
    for t in range(n_obs - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]
    
    return best_path, V

states = [0, 1]
observations = [0, 1, 0]
start_prob = np.array([0.6, 0.4])
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
emit_prob = np.array([[0.1, 0.4], [0.6, 0.3]])

best_path, V = viterbi(observations, states, start_prob, trans_prob, emit_prob)

print("Most likely sequence of states:", best_path)
