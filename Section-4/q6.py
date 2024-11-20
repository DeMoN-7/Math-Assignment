# 6. Develop a Python program to estimate the parameters (transition and emission matrices) of an
# HMM using the Expectation-Maximization algorithm.

import numpy as np
from hmmlearn import hmm

def fit_hmm(data, n_components=2):
    # Reshape data to be 2D as required by hmmlearn
    data = np.array(data).reshape(-1, 1)

    # Create a Gaussian HMM instance
    model = hmm.GaussianHMM(n_components=n_components, n_iter=1000)

    # Fit the model to the data
    model.fit(data)

    # Retrieve the transition and emission matrices
    transition_matrix = model.transmat_  # Transition probabilities
    means = model.means_                  # Emission means
    covariances = model.covars_           # Emission covariances

    return transition_matrix, means, covariances

# Example usage
if __name__ == "__main__":
    # Sample data: a sequence of observations
    sample_data = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0, 1.0]

    # Fit HMM and get parameters
    trans_mat, emission_means, emission_covars = fit_hmm(sample_data)

    print("Transition Matrix:\n", trans_mat)
    print("Emission Means:\n", emission_means)
    print("Emission Covariances:\n", emission_covars)