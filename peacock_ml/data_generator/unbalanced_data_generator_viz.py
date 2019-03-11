import matplotlib.pyplot as plt


def plot_generated_distribution(df_test, df_train, figsize=(10, 7)):
    """
    Display the generated biaised distribution.

    Parameters
    ----------
    df_test: pandas dataframe
    df_train: pandas dataframe
    figsize: tuple, optional (default=(10,7)), set the figure size

    Returns
    -------
    Distribution figure

    Examples
    --------
    from peacock_ml.data_generator.unbalanced_data_generator import create_weighted_multiclass_dataset

    df_train, df_test = create_weighted_multiclass_dataset(n_samples=50000,
                                                           n_classes=5,
                                                           weights=(0.22, 0.08, 0.4, 0.2, 0.1))
    plot_generated_distribution(df_test, df_train)
    """
    y_max = max(df_train.shape[0], df_test.shape[0])
    n_classes = len(df_train.y.unique())

    with plt.xkcd():
        plt.figure(figsize=figsize)
        df_test.y.value_counts().ix[[i for i in range(n_classes)]].plot.bar()
        plt.title('Created test distribution')
        plt.ylim((0, y_max))
        plt.show()
    with plt.xkcd():
        plt.figure(figsize=figsize)
        df_train.y.value_counts().ix[[i for i in range(n_classes)]].plot.bar()
        plt.title('Created train distribution')
        plt.ylim((0, y_max))
        plt.show()
