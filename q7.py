# . Write a Python script to visualize the stationary 
# distribution of a birth-death process using a bar chart.
import numpy as np
import matplotlib.pyplot as plt

def stationary_distribution(birth_rate, death_rate, max_states=100):
    stationary_probs = [1.0]
    for i in range(1, max_states):
        stationary_probs.append(stationary_probs[-1] * (birth_rate / death_rate))
    total_prob = sum(stationary_probs)
    stationary_probs = [p / total_prob for p in stationary_probs]
    return stationary_probs

birth_rate = 0.1
death_rate = 0.05
max_states = 10

stationary_probs = stationary_distribution(birth_rate, death_rate, max_states)

plt.bar(range(max_states), stationary_probs, color='blue')
plt.title("Stationary Distribution of a Birth-Death Process")
plt.xlabel("State")
plt.ylabel("Probability")
plt.show()
