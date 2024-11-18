# . Create a Python function that models a birth-death 
# queueing system and simulates the number of customers over time.

import numpy as np
import matplotlib.pyplot as plt
def birth_death_queue(birth_rate, death_rate, initial_customers, max_time):
    time = [0]
    customers = [initial_customers]
    rng = np.random.default_rng()
    while time[-1] < max_time:
        current_customers = customers[-1]
        birth_prob = birth_rate if current_customers >= 0 else 0
        death_prob = death_rate if current_customers > 0 else 0
        total_rate = birth_prob + death_prob
        if total_rate == 0:
            break
        next_time = time[-1] + rng.exponential(1 / total_rate)
        if rng.uniform() < birth_prob / total_rate:
            next_customers = current_customers + 1
        else:
            next_customers = current_customers - 1
        time.append(next_time)
        customers.append(next_customers)
    return time, customers
birth_rate = 0.5
death_rate = 0.3
initial_customers = 5
max_time = 50
time, customers = birth_death_queue(birth_rate, death_rate, initial_customers, max_time)
plt.figure(figsize=(10, 6))
plt.step(time, customers, where='post')
plt.title("Birth-Death Queueing System Simulation")
plt.xlabel("Time")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.show()
