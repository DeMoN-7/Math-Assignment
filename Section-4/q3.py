# 3. Create a Python function that uses the forward algorithm to 
# calculate the probability of an observed sequence given an HMM.
import numpy as np

def forward_algorithm(observations, start_prob, trans_prob, emit_prob):
    n_states = len(start_prob)
    n_obs = len(observations)
    
    alpha = np.zeros((n_states, n_obs))
    
    # Initialization step
    for state in range(n_states):
        alpha[state, 0] = start_prob[state] * emit_prob[state, observations[0]]
    
    # Recursion step
    for t in range(1, n_obs):
        for state in range(n_states):
            alpha[state, t] = np.sum(alpha[:, t - 1] * trans_prob[:, state]) * emit_prob[state, observations[t]]
    
    # Termination step
    prob = np.sum(alpha[:, n_obs - 1])
    
    return prob

start_prob = np.array([0.6, 0.4])
trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
emit_prob = np.array([[0.1, 0.4], [0.6, 0.3]])
observations = [0, 1, 0]
prob = forward_algorithm(observations, start_prob, trans_prob, emit_prob)
print("Probability of the observed sequence:", prob)
