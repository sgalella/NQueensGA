import numpy as np
from tqdm import tqdm


class MutationTypeError(Exception):
    """
    Selected mutation type does not exist.
    """
    def __init__(self):
        super().__init__("Selected mutation type does not exist.")


class RecombinationTypeError(Exception):
    """
    Selected recombination type does not exist.
    """
    def __init__(self):
        super().__init__("Selected recombination type does not exist.")


class GeneticAlgorithm:
    """
    Genetic algorithm for TSP.
    """
    def __init__(self, board_size=8, num_iterations=1000, population_size=100, offspring_size=20, mutation_rate=0.2,
                 mutation_type="swap", recombination_type="pmx"):
        """
        Initializes the algorithm.
        """
        self.board_size = board_size
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self.recombination_type = recombination_type
        assert self.offspring_size < self.population_size, "Population size has to be greater than the number of selected individuals"

    def __repr__(self):
        """
        Visualizes algorithm parameters when printing.
        """
        return (f"Iterations: {self.num_iterations}\n"
                f"Population size: {self.population_size}\n"
                f"Num selected: {self.num_selected}\n"
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

    @staticmethod
    def recombination_pmx(individual1, individual2, gene1=None, gene2=None):
        """
        Creates a new individual by recombinating two parents using the
        Partially Mapped Crossover (PMX) method.

        Args:
            parent1 (np.array): First parent.
            parent2 (np.array): Second parent.

        Returns:
            new_individual1, new_individual2 (tuple): Recombined individuals.
        """
        # Copy parents
        parent1 = individual1.copy()
        parent2 = individual2.copy()

        # Initialize new individuals
        new_individual1 = -np.ones(len(individual1), dtype=int)
        new_individual2 = -np.ones(len(individual2), dtype=int)

        # Perform the pmx recombination
        # 1. Select two genes at random and copy segment to new individuals
        if gene1 is None or gene2 is None:
            gene1, gene2 = GeneticAlgorithm.choose_random_genes(individual1)
        new_individual1[gene1:gene2 + 1] = parent1[gene1:gene2 + 1]
        new_individual2[gene1:gene2 + 1] = parent2[gene1:gene2 + 1]

        # 2. Replace elements from the segment in the other parent
        for current_gene in range(gene1, gene2 + 1):
            if parent2[current_gene] not in new_individual1:
                pos = np.where(parent2 == parent1[current_gene])[0][0]
                if new_individual1[pos] == -1:
                    new_individual1[pos] = parent2[current_gene]
                else:
                    while new_individual1[pos] != -1:
                        pos = np.where(parent2 == parent1[pos])[0][0]
                    new_individual1[pos] = parent2[current_gene]
            if parent1[current_gene] not in new_individual2:
                pos = np.where(parent2 == parent2[current_gene])[0][0]
                if new_individual2[pos] == -1:
                    new_individual2[pos] = parent1[current_gene]
                else:
                    while new_individual2[pos] != -1:
                        pos = np.where(parent1 == parent2[pos])[0][0]
                    new_individual2[pos] = parent1[current_gene]

        # 3. Complete empty positions with the segment of the opposite parent from 1.
        new_individual1[np.where(new_individual1 == -1)] = parent2[np.where(new_individual1 == -1)]
        new_individual2[np.where(new_individual2 == -1)] = parent1[np.where(new_individual2 == -1)]

        return (new_individual1, new_individual2)

    @staticmethod
    def recombination_order(individual1, individual2, gene1=None, gene2=None):
        """
        Creates a new individual by recombinating two parents using the
        Order Crossover method.

        Args:
            parent1 (np.array): First parent.
            parent2 (np.array): Second parent.

        Returns:
            new_individual1, new_individual2 (tuple): Recombined individuals.
        """
        # Copy parents
        parent1 = individual1.copy()
        parent2 = individual2.copy()

        # Initialize new individuals
        new_individual1 = -np.ones(len(individual1), dtype=int)
        new_individual2 = -np.ones(len(individual2), dtype=int)

        # Perform the order recombination
        # 1. Select two genes at random and copy segment to new individuals
        if gene1 is None or gene2 is None:
            gene1, gene2 = GeneticAlgorithm.choose_random_genes(individual1)
        new_individual1[gene1:gene2 + 1] = parent1[gene1:gene2 + 1]
        new_individual2[gene1:gene2 + 1] = parent2[gene1:gene2 + 1]

        # 2. Fill arrays
        offpring_iterator = gene2 + 1
        while np.count_nonzero(new_individual1 == -1) > 0:
            if np.take(new_individual1, offpring_iterator, mode="wrap") == -1:
                parent_iterator = offpring_iterator
                while np.take(new_individual1, offpring_iterator, mode="wrap") == -1:
                    if np.take(parent2, parent_iterator, mode="wrap") not in new_individual1:
                        new_individual1[offpring_iterator % len(individual1)] = np.take(parent2, parent_iterator, mode="wrap")
                        break
                    parent_iterator += 1
            if np.take(new_individual2, offpring_iterator, mode="wrap") == -1:
                parent_iterator = offpring_iterator
                while np.take(new_individual2, offpring_iterator, mode="wrap") == -1:
                    if np.take(parent1, parent_iterator, mode="wrap") not in new_individual2:
                        new_individual2[offpring_iterator % len(individual2)] = np.take(parent1, parent_iterator, mode="wrap")
                        break
                    parent_iterator += 1
            offpring_iterator += 1

        return (new_individual1, new_individual2)

    @staticmethod
    def recombination_cycle(individual1, individual2, gene1=None, gene2=None):
        """
        Creates a new individual by recombinating two parents using the
        Cycle Crossover method.

        Args:
            parent1 (np.array): First parent.
            parent2 (np.array): Second parent.

        Returns:
            new_individual1, new_individual2 (tuple): Recombined individuals.
        """
        # Copy parents
        parent1 = individual1.copy()
        parent2 = individual2.copy()

        # Initialize offspring
        new_individual1 = individual1.copy()
        new_individual2 = individual2.copy()

        # Perform the cycle recombination
        # 1. Detect cycles
        for position in range(len(individual1)):
            cycle_positions = [position]
            next_position = np.where(parent1 == parent2[position])[0][0]
            while next_position != 0:
                cycle_positions.append(next_position)
                next_position = np.where(parent1 == parent2[next_position])[0][0]
            if len(cycle_positions) < len(individual1):
                break

        # 3. Keep cycle and swap parents positions
        swap_positions = np.array([position for position in range(len(individual1)) if position not in cycle_positions])
        new_individual1[swap_positions] = parent2[swap_positions]
        new_individual2[swap_positions] = parent1[swap_positions]

        return (new_individual1, new_individual2)

    @staticmethod
    def choose_random_genes(individual):
        gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
        while gene2 - gene1 < 2:
            gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
        return (gene1, gene2)

    @staticmethod
    def mutation_swap(individual, gene1=None, gene2=None):
        """
        Mutates indidividual using the swap method.

        Args:
            individual (np.array): Original individual.

        Returns:
            mutated_individual (np.array): Individual mutated.
        """
        if gene1 is None or gene2 is None:
            gene1, gene2 = np.random.choice(len(individual), size=(2, 1), replace=False).flatten()
        mutated_individual = individual.copy()
        mutated_individual[gene1], mutated_individual[gene2] = mutated_individual[gene2], mutated_individual[gene1]
        return mutated_individual

    @staticmethod
    def mutation_insert(individual, gene1=None, gene2=None):
        """
        Mutates indidividual using the insert method.

        Args:
            individual (np.array): Original individual.

        Returns:
            mutated_individual (np.array): Individual mutated.
        """
        if gene1 is None or gene2 is None:
            gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
        mutated_individual = np.concatenate((individual[:gene1 + 1], np.array((individual[gene2],)),
                                             individual[gene1 + 1:gene2], individual[gene2 + 1:]))
        return mutated_individual

    @staticmethod
    def mutation_scramble(individual, gene1=None, gene2=None):
        """
        Mutates indidividual by using the scramble method.

        Args:
            individual (np.array): Original individual.

        Returns:
            mutated_individual (np.array): Individual mutated.
        """
        if gene1 is None or gene2 is None:
            gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
        chromosome = individual[gene1:gene2 + 1]
        chromosome_permuted = np.random.permutation(chromosome)
        while np.array_equal(chromosome, chromosome_permuted):
            chromosome_permuted = np.random.permutation(chromosome)
        mutated_individual = np.concatenate((individual[:gene1], chromosome_permuted, individual[gene2 + 1:]))
        return mutated_individual

    @staticmethod
    def mutation_inversion(individual, gene1=None, gene2=None):
        """
        Mutates indidividual by using the inversion method.

        Args:
            individual (np.array): Individual to be mutated.

        Returns:
            mutated_individual (np.array): Individual mutated.
        """
        if gene1 is None or gene2 is None:
            gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
        chromosome = individual[gene1:gene2 + 1]
        mutated_individual = np.concatenate((individual[:gene1], chromosome[::-1], individual[gene2 + 1:]))
        return mutated_individual

    def generate_next_population(self, population, mutation, recombination):
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

        # Select next generation with probability fitness / total_fitness
        probability_survival = fitness_population / (sum(fitness_population))
        idx_next_population = np.random.choice(range(len(temporal_population)), size=self.population_size, p=probability_survival.flatten())

        return (temporal_population[idx_next_population], fitness_population[idx_next_population])

    def run(self):
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

        # Select mutation
        if self.mutation_type == "swap":
            mutation = self.mutation_swap
        elif self.mutation_type == "insert":
            mutation = self.mutation_insert
        elif self.mutation_type == "scramble":
            mutation = self.mutation_scramble
        elif self.mutation_type == "inversion":
            mutation = self.mutation_inversion
        else:
            raise MutationTypeError

        # Select recombination
        if self.recombination_type == "pmx":
            recombination = self.recombination_pmx
        elif self.recombination_type == "order":
            recombination = self.recombination_order
        elif self.recombination_type == "cycle":
            recombination = self.recombination_cycle
        else:
            raise RecombinationTypeError

        # Iterate through generations
        for iteration in tqdm(range(self.num_iterations), ncols=75):
            population, fitness = self.generate_next_population(population, mutation, recombination)
            best_fitness_iteration = np.max(fitness)
            mean_fitness_iteration = np.mean(fitness)
            diversity_genotype_iteration = np.unique(population, axis=0).shape[0]
            diversity_phenotype_iteration = np.unique(fitness).shape[0]
            max_fitness.append(best_fitness_iteration)
            mean_fitness.append(mean_fitness_iteration)
            diversity_genotype.append(diversity_genotype_iteration)
            diversity_phenotype.append(diversity_phenotype_iteration)
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
