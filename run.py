from n_queens.genetic_algorithm import GeneticAlgorithm
from n_queens.utils import plot_results
from numpy.random import seed

# Set random seed (for reproducibility)
random_seed = 1234
seed(random_seed)

# Initialize parameters
board_size = 10
num_iterations = 1000
population_size = 100
offspring_size = 20
mutation_rate = 0.2
mutation_type = "inversion"

# Initialize genetic algorithm
ga = GeneticAlgorithm(board_size, num_iterations, population_size, offspring_size, mutation_rate, mutation_type)

# Run algorithm
solutions, max_fitness, mean_fitness = ga.run()

# Plot the board
plot_results(solutions[0], mean_fitness, max_fitness)
