# . Implement a Python program that simulates a birth-death process with 
# time-dependent rates, where the birth rate changes with time according to a given function.

import numpy as np
import matplotlib.pyplot as plt

def birth_death_process_time_dependent(birth_rate_func, death_rate, initial_population, max_time):
    time = [0]
    population = [initial_population]
    rng = np.random.default_rng()
    
    while time[-1] < max_time:
        current_population = population[-1]
        birth_rate = birth_rate_func(time[-1])
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
    
    return time, population

def birth_rate_function(t):
    return 0.1 * np.sin(t) + 0.2  # Example time-dependent birth rate function

birth_rate_func = birth_rate_function
death_rate = 0.05
initial_population = 10
max_time = 50

time, population = birth_death_process_time_dependent(birth_rate_func, death_rate, initial_population, max_time)

plt.plot(time, population, color='blue')
plt.title("Birth-Death Process with Time-Dependent Birth Rate")
plt.xlabel("Time")
plt.ylabel("Population Size")
plt.grid(True)
plt.show()
