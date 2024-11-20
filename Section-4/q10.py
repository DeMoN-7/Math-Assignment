# 10. Write a Python function to compute the 
# stationary distribution of an HMM using its transition matrix.
import numpy as np

def compute_stationary_distribution(transition_matrix):
    n_states = transition_matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    
    stationary_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real.flatten()
    stationary_distribution = stationary_vector / stationary_vector.sum()
    
    return stationary_distribution

if __name__ == "__main__":
    transition_matrix = np.array([[0.7, 0.3],
                                   [0.4, 0.6]])
    
    stationary_distribution = compute_stationary_distribution(transition_matrix)
    
    print("Stationary Distribution:", stationary_distribution)