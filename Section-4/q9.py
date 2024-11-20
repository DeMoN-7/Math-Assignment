import numpy as np
import matplotlib.pyplot as plt

def generate_hmm_data(n_samples, n_states, transition_matrix, emission_means, emission_variances):
    states = np.zeros(n_samples, dtype=int)
    observations = np.zeros(n_samples)

    states[0] = np.random.choice(n_states)
    observations[0] = np.random.normal(emission_means[states[0]], np.sqrt(emission_variances[states[0]]))

    for t in range(1, n_samples):
        states[t] = np.random.choice(n_states, p=transition_matrix[states[t - 1]])
        observations[t] = np.random.normal(emission_means[states[t]], np.sqrt(emission_variances[states[t]]))

    return states, observations

def plot_hmm_data(states, observations):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.title('Hidden States')
    plt.plot(states, marker='o', linestyle='-', color='b')
    plt.yticks(range(max(states) + 1), [f'State {i}' for i in range(max(states) + 1)])
    plt.xlabel('Time Step')
    
    plt.subplot(2, 1, 2)
    plt.title('Observations')
    plt.plot(observations, marker='x', linestyle='-', color='r')
    plt.xlabel('Time Step')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_samples = 100
    n_states = 3
    transition_matrix = np.array([[0.7, 0.2, 0.1],
                                   [0.3, 0.4, 0.3],
                                   [0.2, 0.3, 0.5]])
    
    emission_means = [5, 10, 15]
    emission_variances = [1, 1.5, 2]

    states, observations = generate_hmm_data(n_samples, n_states, transition_matrix, emission_means, emission_variances)
    
    plot_hmm_data(states, observations)