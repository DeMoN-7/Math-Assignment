# Write a Python script to perform the Baum-Welch algorithm for training
# an HMM on a given set of observed sequences.
import numpy as np

def forward_algorithm(observations, start_prob, trans_prob, emit_prob, epsilon=1e-10):
    n_states, n_obs = len(start_prob), len(observations)
    alpha = np.full((n_states, n_obs), -np.inf)
    for state in range(n_states):
        alpha[state, 0] = np.log(start_prob[state] + epsilon) + np.log(emit_prob[state, observations[0]] + epsilon)
    
    for t in range(1, n_obs):
        for state in range(n_states):
            alpha[state, t] = np.log(np.sum(np.exp(alpha[:, t - 1] + np.log(trans_prob[:, state] + epsilon)))) + np.log(emit_prob[state, observations[t]] + epsilon)
    
    return alpha

def backward_algorithm(observations, trans_prob, emit_prob, epsilon=1e-10):
    n_states, n_obs = len(trans_prob), len(observations)
    beta = np.full((n_states, n_obs), -np.inf)
    beta[:, n_obs - 1] = 0
    
    for t in range(n_obs - 2, -1, -1):
        for state in range(n_states):
            beta[state, t] = np.log(np.sum(np.exp(np.log(trans_prob[state, :] + epsilon) + np.log(emit_prob[:, observations[t + 1]] + epsilon) + beta[:, t + 1])))
    
    return beta

def baum_welch(observations, n_states, n_iterations, start_prob, trans_prob, emit_prob, epsilon=1e-10):
    n_obs = len(observations)
    for _ in range(n_iterations):
        alpha, beta = forward_algorithm(observations, start_prob, trans_prob, emit_prob, epsilon), backward_algorithm(observations, trans_prob, emit_prob, epsilon)
        
        xi, gamma = np.zeros((n_states, n_states, n_obs - 1)), np.zeros((n_states, n_obs))
        
        for t in range(n_obs - 1):
            denominator = np.log(np.sum(np.exp(alpha[:, t] + np.log(trans_prob + epsilon) + np.log(emit_prob[:, observations[t + 1]] + epsilon) + beta[:, t + 1])))
            for i in range(n_states):
                gamma[i, t] = np.log(np.sum(np.exp(alpha[i, t] + beta[i, t]))) - np.log(np.sum(np.exp(alpha[:, t] + beta[:, t])))
                for j in range(n_states):
                    xi[i, j, t] = alpha[i, t] + np.log(trans_prob[i, j] + epsilon) + np.log(emit_prob[j, observations[t + 1]] + epsilon) + beta[j, t + 1] - denominator
        
        start_prob = np.exp(gamma[:, 0])
        
        for i in range(n_states):
            for j in range(n_states):
                trans_prob[i, j] = np.exp(np.sum(xi[i, j, :])) / np.sum(np.exp(gamma[i, :-1]))
        
        for j in range(n_states):
            for k in range(len(emit_prob[0])):
                emit_prob[j, k] = np.sum(np.exp(gamma[j, observations == k])) / np.sum(np.exp(gamma[j, :]))
    
    return start_prob, trans_prob, emit_prob

observations = np.array([0, 1, 0, 1, 0, 1, 0, 1])
start_prob, trans_prob, emit_prob = np.array([0.6, 0.4]), np.array([[0.7, 0.3], [0.4, 0.6]]), np.array([[0.1, 0.4], [0.6, 0.3]])

start_prob, trans_prob, emit_prob = baum_welch(observations, 2, 100, start_prob, trans_prob, emit_prob)

print("Trained Start Probabilities:", start_prob)
print("Trained Transition Probabilities:", trans_prob)
print("Trained Emission Probabilities:", emit_prob)


