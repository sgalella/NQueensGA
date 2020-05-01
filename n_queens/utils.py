import tkinter as tk
import matplotlib.pyplot as plt


def plot_results(solution, mean_fitness, max_fitness, diversity_genotype, diversity_phenotype):
    """
    Prints the solution board and the algorithm convergence.

    Args:
        solution (np.array): Permutation of rows containing queens.
    """
    # Board window
    root = tk.Tk()
    root.title("N-Queens")
    root.resizable(False, False)
    board_size = len(solution)
    board = [['–' for _ in range(board_size)] for _ in range(board_size)]
    for idx in range(board_size):
        board[solution[idx]][idx] = '*'
    width = 40  # Square size
    image = tk.PhotoImage(file="images/queen.png")
    canvas = tk.Canvas(root, width=board_size * width + 1, height=board_size * width + 1, highlightthickness=False)
    canvas.pack()
    for row in range(board_size):
        for col in range(board_size):
            if (col + row) % 2 == 0:
                canvas.create_rectangle(col * width, row * width, (col + 1) * width, (row + 1) * width, fill="DarkGoldenRod")
            else:
                canvas.create_rectangle(col * width, row * width, (col + 1) * width, (row + 1) * width, fill="LightGoldenRod")
            if board[row][col] == '*':
                canvas.create_image(col * width + width // 2.65, row * width + width // 2, image=image)

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
    plt.show(block=False)

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
    plt.show(block=False)

    root.mainloop()
