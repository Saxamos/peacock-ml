from peacock_ml.biaised_data.biaised_data_handler import find_weights_and_compute_score
from peacock_ml.data_generator.unbalanced_data_generator import create_weighted_multiclass_dataset

df_train, df_test = create_weighted_multiclass_dataset(n_samples=50000,
                                                       n_classes=5,
                                                       weights=(0.22, 0.08, 0.4, 0.2, 0.1))
find_weights_and_compute_score(df_train, df_test)
