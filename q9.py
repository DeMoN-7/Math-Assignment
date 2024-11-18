# Write a Python function that calculates the expected 
# number of individuals at a given time in a birth-death process.

def expected_population(birth_rate, death_rate, initial_population, time):
    return initial_population * np.exp((birth_rate - death_rate) * time)

birth_rate = 0.1
death_rate = 0.05
initial_population = 10
time = 5

expected_pop = expected_population(birth_rate, death_rate, initial_population, time)
print(f"Expected Population at time {time}: {expected_pop}")
