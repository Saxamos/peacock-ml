import unittest

import numpy as np

from peacock_ml.biaised_data.biaised_data_handler import find_weights_and_compute_score
from peacock_ml.data_generator.unbalanced_data_generator import create_weighted_multiclass_dataset

np.random.seed(6)  # set random seed for test


class TestCreateWeightedMulticlassDataset(unittest.TestCase):
    def test_create_weighted_multiclass_dataset_should_return_expected_dataframe(self):
        # Given
        n_samples = 500
        n_classes = 3
        weights = (0.15, 0.65, 0.2)
        df_train, df_test = create_weighted_multiclass_dataset(n_samples, n_classes, weights)

        # When
        weight_by_class_list = find_weights_and_compute_score(df_train, df_test)

        # Then
        expected_weight_by_class_list = [{0: 0.1625, 1: 0.65, 2: 0.1875},
                                         {0: 0.0875, 1: 0.76875, 2: 0.14375},
                                         {0: 0.0625, 1: 0.78125, 2: 0.15625},
                                         {0: 0.05, 1: 0.7875, 2: 0.1625},
                                         {0: 0.0375, 1: 0.778125, 2: 0.184375},
                                         {0: 0.021875, 1: 0.778125, 2: 0.2}]
        self.assertListEqual(weight_by_class_list, expected_weight_by_class_list)
