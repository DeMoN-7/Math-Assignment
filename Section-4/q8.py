# 8. Implement a Python function to compute the likelihood 
# of a given observation sequence using both the forward and backward algorithms.

import numpy as np

def forward_algorithm(observations, initial_probs, transition_matrix, emission_matrix):
    n_states = transition_matrix.shape[0]
    n_observations = len(observations)
    
    alpha = np.zeros((n_observations, n_states))
    alpha[0] = initial_probs * emission_matrix[:, observations[0]]
    
    for t in range(1, n_observations):
        for j in range(n_states):
            alpha[t, j] = np.sum(alpha[t - 1] * transition_matrix[:, j]) * emission_matrix[j, observations[t]]
    
    return np.sum(alpha[-1])

def backward_algorithm(observations, transition_matrix, emission_matrix):
    n_states = transition_matrix.shape[0]
    n_observations = len(observations)
    
    beta = np.zeros((n_observations, n_states))
    beta[-1] = 1
    
    for t in range(n_observations - 2, -1, -1):
        for i in range(n_states):
            beta[t, i] = np.sum(beta[t + 1] * transition_matrix[i] * emission_matrix[:, observations[t + 1]])
    
    return np.sum(beta[0] * initial_probs * emission_matrix[:, observations[0]])

def compute_likelihood(observations, initial_probs, transition_matrix, emission_matrix):
    forward_likelihood = forward_algorithm(observations, initial_probs, transition_matrix, emission_matrix)
    backward_likelihood = backward_algorithm(observations, transition_matrix, emission_matrix)
    
    return forward_likelihood, backward_likelihood

if __name__ == "__main__":
    observations = [0, 1, 0]
    initial_probs = np.array([0.6, 0.4])
    transition_matrix = np.array([[0.7, 0.3],
                                   [0.4, 0.6]])
    emission_matrix = np.array([[0.5, 0.5],
                                 [0.1, 0.9]])
    
    forward_likelihood, backward_likelihood = compute_likelihood(observations, initial_probs, transition_matrix, emission_matrix)
    
    print("Forward Algorithm Likelihood:", forward_likelihood)
    print("Backward Algorithm Likelihood:", backward_likelihood)