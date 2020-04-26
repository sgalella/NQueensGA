from n_queens.genetic_algorithm import GeneticAlgorithm
from n_queens.utils import print_board
from numpy.random import seed
import matplotlib.pyplot as plt

# Set random seed (for reproducibility)
random_seed = 1234
seed(random_seed)

# Initialize parameters
board_size = 12
num_iterations = 1000
population_size = 100
offspring_size = 20
mutation_rate = 0.2
mutation_type = "inversion"

# Initialize genetic algorithm
ga = GeneticAlgorithm(board_size, num_iterations, population_size, offspring_size, mutation_rate, mutation_type)

# Run algorithm
solutions, max_fitness, mean_fitness = ga.run()

for idx, solution in enumerate(solutions):
    print(f"\nSolution {idx}")
    print_board(solution)

# Plot the results
plt.figure()
plt.plot(range(len(mean_fitness)), mean_fitness, 'b')
plt.plot(range(len(max_fitness)), max_fitness, 'r--')
plt.legend(("mean fitness", "max fitness"))
plt.xlabel('iterations')
plt.ylabel('fitness')
plt.title('Fitness convergence')
plt.grid(alpha=0.3)
plt.savefig('images/convergence.jpg')
plt.show()
