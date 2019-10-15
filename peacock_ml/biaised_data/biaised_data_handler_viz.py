import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy


def plot_iteratively_computed_weights(weight_by_class_list, save_gif=False, figsize=(10, 7)):
    """
    Display the computed weights by class at each iteration.

    Parameters
    ----------
    weight_by_class_list: dict of weights
    save_gif: boolean
    figsize: tuple, optional (default=(10,7)), set the figure size

    Returns
    -------
    Iterative approach figure

    Examples
    --------
    from peacock_ml.biaised_data.biaised_data_handler import find_weights_and_compute_score
    from peacock_ml.data_generator.unbalanced_data_generator import create_weighted_multiclass_dataset

    df_train, df_test = create_weighted_multiclass_dataset(n_samples=5000,
                                                           n_classes=4,
                                                           weights=(0.22, 0.48, 0.03, 0.27))
    weight_by_class_list = find_weights_and_compute_score(df_train, df_test)
    plot_iteratively_computed_weights(weight_by_class_list)
    """
    nb_of_class = len(weight_by_class_list[0])
    nb_of_frame = len(weight_by_class_list)
    x = [str(i) for i in range(nb_of_class)]

    fig = plt.figure(figsize=figsize)
    plt.ylim((0, 1))
    plt.title('Weights evolution by iteration')
    plt.ylabel('Proportion')
    plt.xlabel('Class')

    def barlist(i):
        return list(weight_by_class_list[i].values())

    barcollection = plt.bar(x, barlist(0), color=numpy.random.rand(nb_of_class, 3))

    def animate(i):
        y = barlist(i)
        for i, b in enumerate(barcollection):
            b.set_height(y[i])

    anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=nb_of_frame, interval=800)
    if save_gif:
        anim.save('biaised_data.gif', writer='imagemagick', fps=10)
    plt.show()
