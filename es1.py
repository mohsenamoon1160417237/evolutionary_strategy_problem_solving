import numpy as np


def fitness_function(x):
    """The function whose minimum we want to find. x is a list of two values."""
    x1, x2 = x  # Extract the two variables from the list
    return abs(x1 ** 3 - x1 * x2 + x2 ** 2)  # Example function with two variables


def evolutionary_strategy(fitness_function, dimension, population_size, generations, sigma, learning_rate):
    """
    Performs an evolutionary strategy to find the minimum of a function, enforcing constraints.

    Args:
     fitness_function: The function whose minimum we want to find.
     dimension: The number of dimensions of the search space (2 in this case).
     population_size: The number of individuals in the population.
     generations: The number of generations to run the algorithm.
     sigma: The standard deviation of the Gaussian mutation.
     learning_rate: The learning rate for the sigma adaptation.

    Returns:
     The best individual found (a list of its coordinates).
    """

    # Initialize population
    population = [np.clip(np.random.rand(dimension), 0, 1) for _ in range(population_size)]  # Enforce 0-1 range

    # Run evolution
    for generation in range(generations):
        # Evaluate fitness
        fitness_values = [fitness_function(individual) for individual in population]

        # Select parent
        best_index = np.argmin(fitness_values)
        parent = population[best_index]

        # Create offspring
        offspring = []
        for _ in range(population_size):
            new_individual = parent + np.random.normal(scale=sigma, size=dimension)
            new_individual = np.clip(new_individual, 0, 1)  # Enforce 0-1 range
            offspring.append(new_individual)

        # Update population
        population = offspring

        # Adapt sigma
        sigma *= np.exp(learning_rate * (np.mean(fitness_values) - fitness_values[best_index]))

    # Return best individual
    best_individual = population[np.argmin([fitness_function(x) for x in population])]
    return best_individual


# Example usage:
dimension = 2  # Two-dimensional search space
population_size = 100
generations = 200
sigma = 0.1
learning_rate = 0.05

best_individual = evolutionary_strategy(fitness_function, dimension, population_size, generations, sigma, learning_rate)
print("Best individual:", best_individual)
print("Minimum value:", fitness_function(best_individual))
