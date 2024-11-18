# Implement a function in Python that calculates the stationary distribution of a birth-death
# process given birth rate A and death rate μ.
def stationary_distribution(birth_rate, death_rate, max_states=100):
    if birth_rate <= 0 or death_rate <= 0:
        raise ValueError("Birth and death rates must be positive.")
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
print("Stationary Distribution:", stationary_probs)
