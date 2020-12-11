import numpy as np
from tqdm import tqdm

from . import mutation, recombination, selection


class GeneticAlgorithm:
    """
    Genetic algorithm for TSP.
    """
    def __init__(self, board_size=8, population_size=100, offspring_size=20, mutation_rate=0.2,
                 mutation_type=mutation.swap, recombination_type=recombination.pmx, selection_type=selection.genitor):
        """
        Initializes the algorithm.
        """
        self.board_size = board_size
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self.recombination_type = recombination_type
        self.selection_type = selection_type
        assert self.offspring_size < self.population_size, "Population size has to be greater than the number of selected individuals"

    def __repr__(self):
        """
        Visualizes algorithm parameters when printing.
        """
        return (f"Population size: {self.population_size}\n"
                f"Num selected: {self.population_size - self.offspring_size}\n"
                f"Mutation rate: {self.mutation_rate}\n")

    def random_initial_population(self):
        """
        Generates random population of individuals.

        Args:
            num_individuals (int): Number of individuals to be created.

        Returns:
            population (np.array): Population containg the different individuals.
        """
        # Initialize populationN
        population = np.array([np.zeros([self.board_size], dtype=int) for _ in range(self.population_size)])

        # Apply different inhibition to each individual
        for individual in range(self.population_size):
            population[individual] = np.random.permutation(self.board_size)

        return population

    def check_queens(self, individual):
        """
        Checks number of checking queen pairs in the board.

        Args:
            individual (int);

        Returns:
            num_checking (int):
        """
        horizontal_checks = len(individual) - len(set(individual))
        diagonal_checks = 0
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                if individual[j] == individual[i] + j - i or individual[j] == individual[i] - j + i:
                    diagonal_checks += 1
        return (diagonal_checks + horizontal_checks)

    def compute_fitness(self, population):
        """
        Computes the fitness for each individual by calculating the number of checking queens.
        The lesser the number of checking queens, the greater the fitness.

        Args:
            population (np.array): Population containg the different individuals.

        Returns:
            fitness_population (np.array): Fitness of the population.
        """
        fitness_population = np.zeros([len(population), 1])
        for idx, individual in enumerate(population):
            fitness_population[idx] = 1 / (self.check_queens(individual) + 1)
        return fitness_population.flatten()

    def generate_next_population(self, population, mutation, recombination, selection):
        """
        Generates the population for the next iteration.

        Args:
            population (np.array): Population containg the different individuals.

        Returns:
            next_population, fitness_population (tuple): Returns tuple containing the next population and its fitness
        """
        # Initialize new offspring
        offspring = np.array([np.zeros([self.board_size], dtype=int) for _ in range(self.offspring_size)])

        # Recombinate best individuals
        if recombination.__name__ == "edge":
            for individual in range(0, self.offspring_size):
                idx_parent1, idx_parent2 = np.random.choice(self.population_size, size=2, replace=False)
                new_individual1 = recombination(population[idx_parent1], population[idx_parent2])
                offspring[individual] = new_individual1
        else:
            for individual in range(0, self.offspring_size, 2):
                idx_parent1, idx_parent2 = np.random.choice(self.population_size, size=2, replace=False)
                new_individual1, new_individual2 = recombination(population[idx_parent1], population[idx_parent2])
                offspring[individual] = new_individual1
                offspring[individual + 1] = new_individual2

        # Add mutation
        for idx in range(len(population)):
            if np.random.random() < self.mutation_rate:
                individual_mutated = mutation(population[idx])
                population[idx] = individual_mutated

        # Group populations
        temporal_population = np.vstack((population, offspring))
        fitness_population = self.compute_fitness(temporal_population)

        # Select next generation
        survivors = selection(fitness_population)
        survivors = survivors[:self.population_size]

        return (temporal_population[survivors], fitness_population[survivors])

    def run(self, num_iterations):
        """
        Runs the algorithm.

        Returns:
            solutions, max_fitness, mean_fitness (tuple): Returns tuple containing the solutions the fitness mean and max along the iterations
        """
        # Initialize first population
        population = self.random_initial_population()

        # Initialize fitness variables
        mean_fitness = []
        max_fitness = []
        diversity_genotype = []
        diversity_phenotype = []

        # Initialize best_fitness
        best_fitness_all = 0

        # Iterate through generations
        for iteration in tqdm(range(num_iterations), ncols=75):
            population, fitness = self.generate_next_population(population, self.mutation_type, self.recombination_type,
                                                                self.selection_type)

            # Save statistics iteration
            best_fitness_iteration = np.max(fitness)
            mean_fitness_iteration = np.mean(fitness)
            diversity_genotype_iteration = np.unique(population, axis=0).shape[0]
            diversity_phenotype_iteration = np.unique(fitness).shape[0]

            max_fitness.append(best_fitness_iteration)
            mean_fitness.append(mean_fitness_iteration)
            diversity_genotype.append(diversity_genotype_iteration)
            diversity_phenotype.append(diversity_phenotype_iteration)

            # Keep best individuals
            if best_fitness_iteration > best_fitness_all:
                solutions = []
                for best_individual in population[np.where(fitness == best_fitness_iteration)]:
                    if not any((best_individual == individual).all() for individual in solutions):
                        solutions.append(best_individual)
                best_fitness_all = best_fitness_iteration
            elif best_fitness_iteration == best_fitness_all:
                for best_individual in population[np.where(fitness == best_fitness_iteration)]:
                    if not any((best_individual == individual).all() for individual in solutions):
                        solutions.append(best_individual)

        return (np.asarray(solutions), max_fitness, mean_fitness, diversity_genotype, diversity_phenotype)
