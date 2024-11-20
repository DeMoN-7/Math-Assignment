import numpy as np
import matplotlib.pyplot as plt

def create_transition_matrix():
    return np.array([[0.7, 0.3],
                     [0.4, 0.6]])

def plot_transition_diagram(transition_matrix):
    n_states = transition_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(n_states):
        for j in range(n_states):
            if transition_matrix[i, j] > 0:
                ax.annotate('', xy=(j, n_states - i - 1), xytext=(i, n_states - i - 1),
                            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
                ax.text((i + j) / 2, n_states - i - 1 + 0.1, f"{transition_matrix[i, j]:.2f}", 
                         ha='center', fontsize=12)

    ax.set_xticks(range(n_states))
    ax.set_xticklabels([f'State {i}' for i in range(n_states)])
    ax.set_yticks(range(n_states))
    ax.set_yticklabels([f'State {i}' for i in range(n_states)])
    
    ax.set_xlim(-0.5, n_states - 0.5)
    ax.set_ylim(-0.5, n_states - 0.5)
    
    plt.title('State Transition Diagram of HMM')
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    transition_matrix = create_transition_matrix()
    plot_transition_diagram(transition_matrix)