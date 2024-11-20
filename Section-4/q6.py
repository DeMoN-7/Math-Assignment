# 6. Develop a Python program to estimate the parameters (transition and emission matrices) of an
# HMM using the Expectation-Maximization algorithm.

import numpy as np

def em_algorithm(obs, n_states, n_obs, trans_prob, emit_prob, max_iter=100, epsilon=1e-6):
    n_samples = len(obs)
    for iteration in range(max_iter):
        # Initialize alpha, beta, gamma, and xi arrays
        alpha = np.zeros((n_states, n_samples))
        beta = np.zeros((n_states, n_samples))
        gamma = np.zeros((n_states, n_samples))
        xi = np.zeros((n_states, n_states, n_samples - 1))

        # Forward pass
        alpha[:, 0] = emit_prob[:, obs[0]] * trans_prob[:, 0]  # Initial probabilities
        for t in range(1, n_samples):
            for state in range(n_states):
                alpha[state, t] = np.sum(alpha[:, t - 1] * trans_prob[:, state]) * emit_prob[state, obs[t]]

        # Backward pass
        beta[:, n_samples - 1] = 1  # Last time step probabilities
        for t in range(n_samples - 2, -1, -1):
            for state in range(n_states):
                beta[state, t] = np.sum(trans_prob[state, :] * emit_prob[:, obs[t + 1]] * beta[:, t + 1])

        # E-step: Compute gamma and xi
        for t in range(n_samples - 1):
            denom = np.sum(alpha[:, t] * beta[:, t])
            for i in range(n_states):
                gamma[i, t] = (alpha[i, t] * beta[i, t]) / denom
                for j in range(n_states):
                    xi[i, j, t] = (alpha[i, t] * trans_prob[i, j] * emit_prob[j, obs[t + 1]] * beta[j, t + 1]) / denom

        # M-step: Update transition and emission probabilities
        trans_prob = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1)[:, np.newaxis]
        
        for i in range(n_states):
            for j in range(n_obs):
                emit_prob[i, j] = np.sum(gamma[i, obs == j]) / np.sum(gamma[i])

    return trans_prob, emit_prob

# Example usage
obs = np.array([0, 1, 0, 1])  # Observed sequence (e.g., weather states)
n_states = 2                   # Number of hidden states (e.g., sunny or rainy)
n_obs = 2                      # Number of observed states (e.g., dry or wet)

# Initial transition and emission probabilities
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])  # Transition probabilities
emit_prob = np.array([[0.9, 0.1], [0.2, 0.8]])   # Emission probabilities

# Run the EM algorithm to estimate parameters
trans_prob_estimated, emit_prob_estimated = em_algorithm(obs, n_states, n_obs,
                                                          trans_prob.copy(), emit_prob.copy())

print("Estimated Transition Probabilities:")
print(trans_prob_estimated)
print("Estimated Emission Probabilities:")
print(emit_prob_estimated)
