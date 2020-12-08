import numpy as np


def swap(individual, gene1=None, gene2=None):
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


def insert(individual, gene1=None, gene2=None):
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


def scramble(individual, gene1=None, gene2=None):
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


def inversion(individual, gene1=None, gene2=None):
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
