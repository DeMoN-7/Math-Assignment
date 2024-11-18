# . Develop a Python program to simulate multiple runs of a birth-death 
# process and compute the empirical probability distribution of the 
# population size.

import numpy as np
import matplotlib.pyplot as plt

def birth_death_process(birth_rate, death_rate, initial_population, max_time):
    rng = np.random.default_rng()
    population = [initial_population]
    time = [0]
    
    while time[-1] < max_time:
        current_population = population[-1]
        total_rate = birth_rate * current_population + death_rate * current_population
        
        if total_rate == 0:
            break
        
        next_time = time[-1] + rng.exponential(1 / total_rate)
        if rng.uniform() < (birth_rate * current_population) / total_rate:
            next_population = current_population + 1
        else:
            next_population = current_population - 1
        
        time.append(next_time)
        population.append(next_population)
    
    return population

def empirical_distribution(birth_rate, death_rate, initial_population, max_time, num_runs):
    populations = []
    
    for _ in range(num_runs):
        populations.extend(birth_death_process(birth_rate, death_rate, initial_population, max_time))
    
    unique, counts = np.unique(populations, return_counts=True)
    probabilities = counts / len(populations)
    
    return unique, probabilities

birth_rate = 0.1
death_rate = 0.05
initial_population = 5
max_time = 100
num_runs = 1000

states, probabilities = empirical_distribution(birth_rate, death_rate, initial_population, max_time, num_runs)

plt.bar(states, probabilities, color='blue')
plt.title("Empirical Probability Distribution of Population Size")
plt.xlabel("Population Size")
plt.ylabel("Empirical Probability")
plt.show()

