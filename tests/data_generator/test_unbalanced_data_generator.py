import unittest

import numpy as np
import pandas as pd

from peacock_ml.data_generator.unbalanced_data_generator import create_weighted_multiclass_dataset

np.random.seed(6)  # set random seed for test


class TestCreateWeightedMulticlassDataset(unittest.TestCase):
    def test_create_weighted_multiclass_dataset_should_return_expected_dataframe(self):
        # Given
        n_samples = 10
        n_classes = 2
        weights = (0.2, 0.8)

        # When
        df_train, df_test = create_weighted_multiclass_dataset(n_samples, n_classes, weights, random_state=6)

        # Then
        expected_df_train = pd.DataFrame({0: [0.162562, -0.873337],
                                          1: [-1.130677, -1.038740],
                                          2: [0.239788, 2.198986],
                                          3: [-1.014991, -0.203573],
                                          4: [1.134153, 1.335500],
                                          5: [1.345242, 1.545565],
                                          'y': [0, 1]}, index=[7, 8])
        expected_df_test = pd.DataFrame(
            {0: [-0.202905, -0.598007, -0.299034, -0.435377, -0.528892, 0.132532, -0.858368, 0.403904],
             1: [-0.337185, 0.533608, -0.441403, -1.136436, 1.116964, 1.684785, 2.612718, -0.127225],
             2: [1.177117, 0.411665, 0.311934, 1.309396, 1.219284, -0.568842, 1.651565, -0.899655],
             3: [-1.022236, -1.107667, -0.972988, -0.671509, -0.680816, -0.483823, -0.402107, -0.641576],
             4: [0.879583, 1.384150, 1.414394, 1.411274, 0.244817, - 0.618407, - 0.518109, 0.293336],
             5: [0.4489401516713166, -0.054485402010911246, 0.8882502756286543, 1.5689878106413613, -1.1209963491997077,
                 -1.993460114356073, -2.833076894937894, 0.12029903882849674],
             'y': [0, 1, 1, 1, 1, 1, 1, 1, ]}, index=[4, 0, 1, 2, 3, 5, 6, 9]
        )
        pd.testing.assert_frame_equal(df_train, expected_df_train)
        pd.testing.assert_frame_equal(df_test, expected_df_test)
