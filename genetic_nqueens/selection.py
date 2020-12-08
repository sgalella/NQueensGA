import numpy as np


def genitor(fitness_population):
    """
    Selects population using the genitor method.

    Args:
        population (np.array): Population containg the different individuals.
        fitness_population (np.array): Fitness of the population.
    """
    survivors = np.argsort(fitness_population)
    return survivors[::-1]