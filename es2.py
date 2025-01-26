import random
import numpy as np


def ackley(x: list):
    """
    Ackley function
    """
    first_term = -20 * np.exp(-0.2 * np.sqrt(1 / 30) * sum([x[n] ** 2 for n in range(30)]))
    second_term = np.exp(1 / 30 * sum([np.cos(2 * np.pi * x[m]) for m in range(30)]))
    return first_term - second_term + np.exp(1) + 20


def generate_population(size, lower_bound, upper_bound):
    """
    Generates a population of individuals
    """
    population = []
    for _ in range(size):
        individual = [random.uniform(lower_bound, upper_bound) for _ in range(30)]  # 2 dimensions for Ackley
        population.append(individual)
    return population


def main():
    """
    Main function for the optimization process
    """

    # Parameters
    population_size_array = [30, 200]
    # population_size = 30
    iterations = 2000
    lower_bound = -40
    upper_bound = 40
    sigma = 0.1
    dimension = 30
    learning_rate = 0.1

    # Initialize the population
    population_size = population_size_array[0]
    population_size_array = population_size_array[::-1]
    population = generate_population(population_size, lower_bound, upper_bound)
    fitness_values = [ackley(individual) for individual in population]

    # Main loop
    for i in range(iterations):
        # Evaluate fitness (Ackley function)


        # Select the best individuals (no explicit selection in this strategy)

        # Mutation

        best_index = np.argmin(fitness_values)
        parent = population[best_index]

        # Create offspring
        offspring = []
        for _ in range(population_size):
            new_individual = parent + np.random.normal(scale=sigma, size=dimension)
            new_individual = np.clip(new_individual, lower_bound, upper_bound)  # Enforce -40 to 40 range
            offspring.append(new_individual)

        # Update population
        population = offspring

        # Adapt sigma
        sigma *= np.exp(learning_rate * (np.mean(fitness_values) - fitness_values[best_index]))
        # Update the population size
        population_size = population_size_array[0]
        population_size_array = population_size_array[::-1]
        # population = population[:population_size]  # Keep only the required number

        # Print best individual in each generation
        fitness_values = [ackley(individual) for individual in population]
        best_index = np.argmin(fitness_values)
        print(f"Generation {i + 1}: Best individual = {population[best_index]}, Fitness = {fitness_values[best_index]}")

    # Find the minimum value and its location after all generations
    min_value = min(fitness_values)
    min_index = np.argmin(fitness_values)
    print(f"\nFinal Result: Minimum value = {min_value} at x = {population[min_index]}")


if __name__ == "__main__":
    main()
