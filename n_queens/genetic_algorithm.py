import numpy as np
from tqdm import tqdm


class GeneticAlgorithm:
    """
    Genetic algorithm for TSP.
    """
    def __init__(self, board_size=8, num_iterations=1000, population_size=100, offspring_size=20, mutation_rate=0.2):
        """
        Initializes the algorithm.
        """
        self.board_size = board_size
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        assert self.offspring_size < self.population_size, "Population size has to be greater than the number of selected individuals"

    def __repr__(self):
        """
        Visualizes algorithm parameters when printing.
        """
        return (f"Iterations: {self.num_iterations}\n"
                f"Population size: {self.population_size}\n"
                f"Num selected: {self.num_selected}\n"
                f"Mutation rate: {self.mutation_rate}\n")

    def random_population(self, num_individuals):
        """
        Generates random population of individuals

        Args:
            num_individuals (int): Number of individuals to be created.

        Returns:
            population (np.array): Population containg the different individuals.
        """
        # Initialize populationN
        population = np.array([np.zeros([self.board_size], dtype=int) for _ in range(num_individuals)])

        # Apply different inhibition to each individual
        for individual in range(num_individuals):
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
        Computes the fitness for each individual by calculating the distances of the cities.

        Args:
            population (np.array): Population containg the different individuals.

        Returns:
            fitness_population (np.array): Fitness of the population.
        """
        fitness_population = np.zeros([len(population), 1])
        for idx, individual in enumerate(population):
            fitness_population[idx] = 1 / (self.check_queens(individual) + 1)

        return fitness_population.flatten()

    def recombination(self, parent1, parent2):
        """
        Creates a new individual by recombinating two parents.

        Args:
            parent1 (np.array): First parent.
            parent2 (np.array): Second parent.

        Returns:
            new_individual: Recombinated individual.
        """
        crossover_points = np.random.randint(self.board_size)
        new_individual1 = np.concatenate((parent1[:crossover_points], parent2[crossover_points:]))
        new_individual2 = np.concatenate((parent2[:crossover_points], parent1[crossover_points:]))
        return [new_individual1, new_individual2]

    def mutation(self, individual):
        """
        Mutates indidividual by changing the position of different cities.

        Args:
            individual (np.array): Individual to be mutated.
        """
        gene1, gene2 = np.random.choice(self.board_size, size=(2, 1), replace=False)
        individual[gene1], individual[gene2] = individual[gene2], individual[gene1]

    def generate_next_population(self, population):
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
        for individual in range(0, self.offspring_size, 2):
            idx_parent1, idx_parent2 = np.random.choice(self.population_size, size=2, replace=False)
            new_individual1, new_individual2 = self.recombination(population[idx_parent1], population[idx_parent2])
            offspring[individual] = new_individual1
            offspring[individual + 1] = new_individual2

        # Add mutation
        for idx in range(len(population)):
            if np.random.random() < self.mutation_rate:
                self.mutation(population[idx])

        temporal_population = np.vstack((population, offspring))
        fitness_population = self.compute_fitness(temporal_population)
        probability_survival = fitness_population / (sum(fitness_population))
        idx_next_population = np.random.choice(range(len(temporal_population)), size=self.population_size, p=probability_survival.flatten())

        return (temporal_population[idx_next_population], fitness_population[idx_next_population])

    def run(self):
        """
        Runs the algorithm.

        Args:
            iter_info (int, optional): Frequency of information in screen. Defaults to 10.

        Returns:
            solutions, max_fitness, mean_fitness (tuple): Returns tuple containing the solutions the fitness mean and max along the iterations
        """
        # Initialize first population
        population = self.random_population(self.population_size)

        # Initialize fitness variables
        mean_fitness = []
        max_fitness = []

        # Initialize best_fitness
        best_fitness_all = 0

        # Iterate through generations
        for iteration in tqdm(range(self.num_iterations), ncols=75):
            population, fitness = self.generate_next_population(population)
            best_fitness_iteration = np.max(fitness)
            mean_fitness_iteration = np.mean(fitness)
            max_fitness.append(best_fitness_iteration)
            mean_fitness.append(mean_fitness_iteration)
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

        return (np.asarray(solutions), max_fitness, mean_fitness)
