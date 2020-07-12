from genetic_nqueens import GeneticAlgorithm, plot_results
from numpy.random import seed

# Set random seed (for reproducibility)
random_seed = 1234
seed(random_seed)

# Initialize parameters
board_size = 10
num_iterations = 1000
population_size = 100
offspring_size = 20
mutation_rate = 0.05
mutation_type = "inversion"
recombination_type = "edge"
selection_type = "genitor"

# Initialize genetic algorithm
ga = GeneticAlgorithm(board_size, num_iterations, population_size, offspring_size, mutation_rate, mutation_type, recombination_type)

# Run algorithm
solutions, max_fitness, mean_fitness, diversity_genotype, diversity_phenotype = ga.run()

# Plot the board
plot_results(solutions[0], mean_fitness, max_fitness, diversity_genotype, diversity_phenotype)
