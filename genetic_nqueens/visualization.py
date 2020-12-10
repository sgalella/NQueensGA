import matplotlib.pyplot as plt
from matplotlib import colors


def plot_board(solution):
    # Board window
    _, ax = plt.subplots()
    board_size = len(solution)
    cmap = colors.LinearSegmentedColormap.from_list(None, ['DarkGoldenRod', 'Gold', 'LimeGreen', 'Red'])
    norm = plt.Normalize(0, 3)  # 4 colors to represent the board
    board = [[1 if (row + col) % 2 == 0 else 0 for col in range(board_size)] for row in range(board_size)]
    wrong_queens = _get_wrong_positions(solution)
    for idx in range(board_size):
        if idx not in wrong_queens:
            board[solution[idx]][idx] = 2
        else:
            board[solution[idx]][idx] = 3
    ax.imshow(board, cmap=cmap, norm=norm)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('images/board.jpg')
    plt.draw()


def plot_fitness(mean_fitness, max_fitness):
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


def plot_diversity(diversity_genotype, diversity_phenotype):
    plt.figure()
    plt.bar(range(len(diversity_phenotype)), diversity_genotype, color='lime', alpha=0.5)
    plt.bar(range(len(diversity_phenotype)), diversity_phenotype, color='orange', alpha=0.5)
    plt.legend(("genotype diversity", "phenotype diversity"))
    plt.xlabel('iterations')
    plt.ylabel('diversity')
    plt.title('Diversity through generations')
    plt.grid(alpha=0.3)
    plt.savefig('images/diversity.jpg')
    plt.draw()


def _get_wrong_positions(solution):
    """
    Returns the positions of queens that are incompatible.

    Args:
        solution (np.array): Row position of each queen in their respective column.

    Returns:
        set: Wrong queen positions.
    """
    wrong = set()
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if solution[i] == solution[j]:
                wrong.update([i, j])
            else:
                if abs(solution[i] - solution[j]) == abs(i - j):
                    wrong.update([i, j])
    return wrong
