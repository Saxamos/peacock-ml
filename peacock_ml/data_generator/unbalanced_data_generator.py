import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

PERCENTAGE_TO_KEEP_IN_TRAIN = 0.66


def create_weighted_multiclass_dataset(
    n_samples, n_classes, weights, random_state=None
):
    """
    A function that generate unbalanced dataset for classification.

        Parameters
        ----------
        n_samples: The number of observation to generate.
        n_classes: The number of classes.
        weights: The proportion of each class in the global generated dataset.
         Note that train and test dataset are subsample so the distributions will be different.
         The sum must be equal to 1.
        random_state: (default None) Choose a seed

        Returns
        -------
        df_train, df_test: Two pandas dataframe

        Examples
        --------
        df_train, df_test = create_weighted_multiclass_dataset(n_samples=50000,
                                                               n_classes=5,
                                                               weights=(0.22, 0.08, 0.4, 0.2, 0.1))
    """
    assert len(weights) == n_classes, "Invalid number of weights."
    X, y = make_classification(
        n_classes=n_classes,
        n_informative=4,
        n_clusters_per_class=1,
        n_samples=n_samples,
        n_features=6,
        flip_y=0.10,
        class_sep=0.7,
        weights=weights,
        random_state=random_state,
    )
    df = pd.DataFrame(X)
    df["y"] = y

    indexes_by_class = _select_indexes_by_class(df, n_classes)
    number_of_train_by_class = _compute_number_of_sample_for_each_train_classes(df)
    test_indexes, train_indexes = _compute_train_and_test_indexes(
        indexes_by_class, number_of_train_by_class
    )

    df_train = df.loc[np.concatenate(train_indexes)]
    df_test = df.loc[np.concatenate(test_indexes)]

    return df_train, df_test


def _compute_train_and_test_indexes(indexes_by_class, number_of_train_by_class):
    train_indexes = [
        np.random.choice(index, number_of_train_by_class, replace=False)
        for index in indexes_by_class
    ]
    test_indexes = [
        list(set(index) - set(train_index))
        for index, train_index in zip(indexes_by_class, train_indexes)
    ]
    return test_indexes, train_indexes


def _compute_number_of_sample_for_each_train_classes(df):
    least_represented_class = df["y"].value_counts().idxmin()
    number_of_train_by_class = int(
        df[df.y == least_represented_class].index.shape[0] * PERCENTAGE_TO_KEEP_IN_TRAIN
    )
    print(f"number_of_train_by_class: {number_of_train_by_class}")
    return number_of_train_by_class


def _select_indexes_by_class(df, n_classes):
    return [df[df.y == i].index for i in range(n_classes)]
