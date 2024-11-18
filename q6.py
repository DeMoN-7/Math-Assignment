# . Implement a Python function to find the mean and variance of the population
# size in a birth- death process at steady state.

def mean_variance_steady_state(birth_rate, death_rate):
    if birth_rate == death_rate: return None, None
    mean = birth_rate / (death_rate - birth_rate)
    variance = birth_rate / ((death_rate - birth_rate)**2)
    return mean, variance

birth_rate, death_rate = 0.1, 0.05
mean, variance = mean_variance_steady_state(birth_rate, death_rate)
print(f"Mean: {mean}, Variance: {variance}")
