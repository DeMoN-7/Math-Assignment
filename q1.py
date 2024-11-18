# Write a Python program to simulate a birth-death process with specified 
# birth and death rates, and plot the population size over time.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
birth_rate = 0.1
death_rate = 0.05
time_steps = 1000
initial_population = 50
scale_factor = 1e5  # Scale to prevent overflow for large populations

# Use the new random generator API
rng = np.random.default_rng()

# Initialize population and time arrays
population = [initial_population]
time = [0]

# Simulate the birth-death process
for t in range(1, time_steps + 1):
    current_population = population[-1]
    
    # Scale population if it gets too large
    scaled_population = min(current_population, scale_factor)
    births = rng.binomial(int(scaled_population), birth_rate)
    deaths = rng.binomial(int(scaled_population), death_rate)
    
    # Scale results back to actual population size
    births = int(births * (current_population / scaled_population))
    deaths = int(deaths * (current_population / scaled_population))
    
    # Update population
    new_population = current_population + births - deaths
    new_population = max(0, new_population)  # Population cannot be negative
    
    # Store results
    population.append(new_population)
    time.append(t)

# Plot the population size over time
plt.figure(figsize=(10, 6))
plt.plot(time, population, label="Population Size", color="blue")
plt.title("Birth-Death Process Simulation", fontsize=16)
plt.xlabel("Time Steps", fontsize=14)
plt.ylabel("Population Size", fontsize=14)
plt.axhline(initial_population, color='red', linestyle='--', label="Initial Population")
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
