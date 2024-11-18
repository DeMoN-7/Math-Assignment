# 5. Implement a Python function that computes the backward probabilities for an HMM 
# and a given observation sequence.
import numpy as np

def backward_algorithm(observations, trans_prob, emit_prob, epsilon=1e-10):
    n_states, n_obs = len(trans_prob), len(observations)
    beta = np.full((n_states, n_obs), -np.inf)
    beta[:, n_obs - 1] = 0  # At the last time step, beta = 0
    
    for t in range(n_obs - 2, -1, -1):
        for state in range(n_states):
            beta[state, t] = np.log(np.sum(np.exp(np.log(trans_prob[state, :] + epsilon) + np.log(emit_prob[:, observations[t + 1]] + epsilon) + beta[:, t + 1])))
    
    return beta

# Example Usage:
observations = np.array([0, 1, 0, 1, 0, 1, 0, 1])
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])  # Transition probabilities
emit_prob = np.array([[0.1, 0.4], [0.6, 0.3]])  # Emission probabilities

beta = backward_algorithm(observations, trans_prob, emit_prob)
print("Backward Probabilities:")
print(beta)
