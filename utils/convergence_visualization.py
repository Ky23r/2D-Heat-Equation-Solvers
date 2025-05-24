import matplotlib.pyplot as plt


def plot_convergence_curves(*error_histories, labels=None, max_iter=400):
    num_curves = len(error_histories)
    if labels is None:
        labels = [f"Curve {i+1}" for i in range(num_curves)]
    elif len(labels) != num_curves:
        raise ValueError("Number of labels must match number of error histories")

    plt.figure()
    for error_history, label in zip(error_histories, labels):
        iters_to_plot = min(len(error_history), max_iter)
        plt.plot(range(iters_to_plot), error_history[:iters_to_plot], label=label)

    plt.xlabel("Iteration Number")
    plt.ylabel("Maximum Absolute Error")
    plt.title(
        f"Convergence Behavior: Maximum Absolute Error Over Iterations\n(First {max_iter} Iterations)"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
