import sklearn as sk
import xgboost as xgb

NUMBER_OF_ITERATION = 4


def find_weights_and_compute_score(df_train, df_test, epsilon=0.01):
    """
    A function that finds the weights with an iterative approach in order to improve model when there is a biais in
    data distribution, i.e: P[Y|X] is not the same in train and test set.

        Parameters
        ----------
        df_train: train dataframe.
        df_test: test dataframe.
        epsilon: float

        Returns
        -------
        weight_by_class: A dictionary of weight by class

        References
        ----------
        [1]  Shimodaira et al., "Improving predictive inference under covariate shift by weighting the log-likelihood
        function", Journal of statistical planning and inference, 2000.

        [2]  A. de Myttenaere, "Machine learning using biaised data", ML Meetup Sydney, 2017.
    """
    dtrain_eval, eval_list = _split_train_for_early_stopping(df_train)
    xgb_params = {'max_depth': 4, 'eta': 1, 'silent': 1, 'num_class': len(df_train.y.unique()),
                  'objective': 'multi:softmax', 'eval_metric': 'merror'}
    best_iteration = _find_best_iteration(dtrain_eval, eval_list, xgb_params)

    initial_weights = _compute_equal_weights_for_each_class(df_train)

    predictions = _make_prediction(df_train, df_test, initial_weights, xgb_params, best_iteration)
    weight_by_class = _compute_weight_by_class(df_train, predictions)
    print('Initialization done.\n')
    weights = [weight_by_class[instance] for instance in df_train.y]

    current_iteration, delta = 0, epsilon + 1
    while current_iteration <= NUMBER_OF_ITERATION and delta > epsilon:
        predictions = _make_prediction(df_train, df_test, weights, xgb_params, best_iteration)
        new_weight_by_class = _compute_weight_by_class(df_train, predictions)
        delta = _compute_delta(new_weight_by_class, weight_by_class)
        weights = [new_weight_by_class[instance] for instance in df_train.y]
        current_iteration += 1

    return new_weight_by_class


def _compute_delta(new_weight_by_class, weight_by_class):
    delta_array = [abs(new_weight_by_class[cl] - weight_by_class[cl]) for cl in weight_by_class.keys()]
    delta = sum(delta_array) / len(delta_array)
    print('Delta:', delta, '\n')
    return delta


def _compute_equal_weights_for_each_class(df_train):
    return [1 / len(df_train.y.unique()) for _ in range(df_train.shape[0])]


def _compute_weight_by_class(df_train, preds):
    weight_by_class = {i: (preds == i).mean() for i in df_train.y.unique()}
    print('Weight by class:', weight_by_class)
    return weight_by_class


def _make_prediction(df_train, df_test, weights, xgb_params, best_iteration):
    dtrain = xgb.DMatrix(df_train.iloc[:, :-1].values, df_train.y.values, weight=weights)
    dtest = xgb.DMatrix(df_test.iloc[:, :-1].values)
    clf = xgb.train(xgb_params, dtrain, num_boost_round=best_iteration)
    predictions = clf.predict(dtest)
    print('Accuracy score:', sk.metrics.accuracy_score(df_test.y, predictions))
    return predictions


def _find_best_iteration(dtrain_eval, eval_list, xgb_params):
    eval_clf = xgb.train(xgb_params,
                         dtrain_eval,
                         num_boost_round=1000,
                         evals=eval_list,
                         early_stopping_rounds=30,
                         verbose_eval=False)
    return eval_clf.best_iteration


def _split_train_for_early_stopping(df_train):
    X_train, X_eval, y_train, y_eval = sk.model_selection.train_test_split(df_train.iloc[:, :-1].values,
                                                                           df_train.y.values,
                                                                           test_size=0.3,
                                                                           random_state=6)
    dtrain_eval = xgb.DMatrix(X_train, y_train)
    deval = xgb.DMatrix(X_eval, y_eval)
    eval_list = [(deval, 'eval'), (dtrain_eval, 'train')]
    return dtrain_eval, eval_list