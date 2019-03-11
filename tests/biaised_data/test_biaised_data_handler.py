import unittest

import numpy as np

from peacock_ml.biaised_data.biaised_data_handler import find_weights_and_compute_score
from peacock_ml.data_generator.unbalanced_data_generator import create_weighted_multiclass_dataset

np.random.seed(6)  # set random seed for test


class TestCreateWeightedMulticlassDataset(unittest.TestCase):
    def test_create_weighted_multiclass_dataset_should_return_expected_dataframe(self):
        # Given
        n_samples = 100
        n_classes = 2
        weights = (0.35, 0.65)
        df_train, df_test = create_weighted_multiclass_dataset(n_samples, n_classes, weights)

        # When
        weights = find_weights_and_compute_score(df_train, df_test)

        # Then
        self.assertDictEqual(weights, {0: 0.3333333333333333, 1: 0.6666666666666666})
