import matplotlib.pyplot as plt
from matplotlib import colors


def plot_results(solution, mean_fitness, max_fitness, diversity_genotype, diversity_phenotype):
    """
    Prints the solution board and the algorithm convergence.

    Args:
        solution (np.array): Permutation of rows containing queens.
    """
    # Board window
    _, ax = plt.subplots()
    board_size = len(solution)
    cmap = colors.ListedColormap(['DarkGoldenRod', 'Gold', 'LimeGreen'])
    board = [[1 if (row + col) % 2 == 0 else 0 for col in range(board_size)] for row in range(board_size)]
    for idx in range(board_size):
        board[solution[idx]][idx] = 2
    ax.imshow(board, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('images/board.jpg')
    plt.draw()

    # Convergence figure
    plt.figure()
    plt.plot(range(len(mean_fitness)), mean_fitness, 'b')
    plt.plot(range(len(max_fitness)), max_fitness, 'r--')
    plt.legend(("mean fitness", "max fitness"))
    plt.xlabel('iterations')
    plt.ylabel('fitness')
    plt.title('Fitness through generations')
    plt.grid(alpha=0.3)
    plt.savefig('images/convergence.jpg')
    plt.draw()

    # Diversity bar plot
    plt.figure()
    plt.bar(range(len(diversity_phenotype)), diversity_genotype, color='lime', alpha=0.5)
    plt.bar(range(len(diversity_phenotype)), diversity_phenotype, color='orange', alpha=0.5)
    plt.legend(("genotype diversity", "phenotype diversity"))
    plt.xlabel('iterations')
    plt.ylabel('diversity')
    plt.title('Diversity through generations')
    plt.grid(alpha=0.3)
    plt.savefig('images/diversity.jpg')
    plt.show()


def find_checked_queens(solution):
    pass
