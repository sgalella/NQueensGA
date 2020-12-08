import numpy as np


def pmx(individual1, individual2, gene1=None, gene2=None):
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
        gene1, gene2 = choose_random_genes(individual1)
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


def edge(individual1, individual2, current_element=None):
    """
    Creates a new individual by recombinating two parents using the
    Edge Crossover method.

    Args:
        individual1 (np.array): First parent.
        individual1 (np.array): Second parent.

    Returns:
        new_individual1(np.array): Recombined individual.
    """
    # Copy parents
    parent1 = individual1.copy()
    parent2 = individual2.copy()

    # Create adjacency dictionary
    adjacency = create_adjecency(parent1, parent2)

    # Initialize new individual as an empty list
    new_individual = []
    if current_element is None:
        current_element = np.random.choice(np.array(list(adjacency.keys())))
    new_individual.append(current_element)

    while len(new_individual) < len(parent1):
        # Remove edges from current element
        current_element_edges = adjacency[current_element]
        adjacency = delete_edges(adjacency, current_element)
        adjacency.pop(current_element)

        # Select next gene and append it to the offspring
        current_element = select_next_element(adjacency, current_element_edges)
        new_individual.append(current_element)

    return np.array(new_individual)


def order(individual1, individual2, gene1=None, gene2=None):
    """
    Creates a new individual by recombinating two parents using the
    Order Crossover method.

    Args:
        individual1 (np.array): First parent.
        individual1 (np.array): Second parent.

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
        gene1, gene2 = choose_random_genes(individual1)
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


def cycle(individual1, individual2, gene1=None, gene2=None):
    """
    Creates a new individual by recombinating two parents using the
    Cycle Crossover method.

    Args:
        individual1 (np.array): First parent.
        individual1 (np.array): Second parent.

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


def create_adjecency(individual1, individual2):
    """
    Creates adjacency dictionary by inspecting neighbors of different genes.
    Genotype wraps up in the borders.

    Args:
        individual1 (np.array): First parent.
        individual1 (np.array): Second parent.

    Returns:
        adjacency (dict): Description of edges in the genotype of ascendants.
    """
    adjacency = {}
    # Parent 1
    for idx, element in enumerate(individual1):
        if idx == 0:
            adjacency[element] = [individual1[-1], individual1[idx + 1]]
        elif idx == len(individual1) - 1:
            adjacency[element] = [individual1[idx - 1], individual1[0]]
        else:
            adjacency[element] = [individual1[idx - 1], individual1[idx + 1]]
    # Parent2
    for idx, element in enumerate(individual2):
        if idx == 0:
            adjacency[element].extend([individual2[-1], individual2[idx + 1]])
        elif idx == len(individual2) - 1:
            adjacency[element].extend([individual2[idx - 1], individual2[0]])
        else:
            adjacency[element].extend([individual2[idx - 1], individual2[idx + 1]])
    return adjacency


def delete_edges(adjacency, current_element):
    """
    Removes current_element from edges in adjacency.

    Args:
        adjacency (dict): Description of edges in the genotype of ascendants.
        current_element_edges (list): Edges of the current element.

    Returns:
        adjacency (dict): Updated adjacency without edges of current element.
    """
    for key in adjacency.keys():
        while current_element in adjacency[key]:
            adjacency[key].remove(current_element)
    return adjacency


def select_next_element(adjacency, current_element_edges):
    """
    Selects the next edge by inspecting the edges of the current element.
    Preference is as follows: repeated edge, shortes list, random.

    Args:
        adjacency (dict): Description of edges in the genotype of ascendants.
        current_element_edges (list): Edges of the current element.

    Returns:
        next_current_element: Next gene to include in the offspring.
    """
    if not current_element_edges:
        return np.random.choice(list(adjacency.keys()))
    edge_length = 4 * np.ones(len(current_element_edges))  # Max number of different neighbors
    for idx, element in enumerate(current_element_edges):
        if current_element_edges.count(element) > 1:
            return element
        else:
            edge_length[idx] = len(np.unique(adjacency[element]))

    # Break ties randomly
    min_edge_length = np.random.choice(np.where(edge_length == edge_length.min())[0])

    return current_element_edges[min_edge_length]


def choose_random_genes(individual):
    """
    Selects two separate genes from individual.

    Args:
        individual (np.array): Genotype of individual.

    Returns:
        gene1, gene2 (tuple): Genes separated by at least another gene.
    """
    gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
    while gene2 - gene1 < 2:
        gene1, gene2 = np.sort(np.random.choice(len(individual), size=(2, 1), replace=False).flatten())
    return (gene1, gene2)
