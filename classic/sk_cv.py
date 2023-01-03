import sklearn 
import sklearn.model_selection
import sklearn, sklearn.preprocessing, sklearn.pipeline, sklearn.model_selection
import sklearn.svm, sklearn.tree, sklearn.ensemble, sklearn.neural_network
import sklearn.linear_model, sklearn.gaussian_process, sklearn.neighbors
import sklearn.multioutput
import numpy

import xgboost

from collections import Counter


def cv_trial(training_X, training_y, model_cfgs):
    """My simple cross validation_loop 
    """
    estimator, hyperparam_grid = model_cfgs['estimator'], model_cfgs['param_grid']
    
    search = sklearn.model_selection.GridSearchCV(
            estimator = estimator, 
            param_grid = hyperparam_grid, 
            # scoring = ['neg_root_mean_squared_error'],
            # refit = 'neg_root_mean_squared_error',
            refit = True, 
            scoring = 'balanced_accuracy',
            cv = sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0), 
            n_jobs=10,
            verbose=0
        )

    # run grid search
    search.fit(training_X, training_y)

    return search.best_params_, search.best_score_


test_cfgs = {
#     "nn":{
#         'estimator': sklearn.neural_network.MLPClassifier(shuffle=True),
# #            Test grid
#         'param_grid':   {
#             # hidden_layer_sizes made the search space many order of magnitudes larger
#             'activation':         ['tanh', 'logistic', 'relu'], 
#             'max_iter':           [400*i for i in range(1, 2)], 
#             'learning_rate':      ['adaptive']
#         }
#     },
    "xgboost" : {
        'estimator': xgboost.XGBClassifier(),
        'param_grid': {
            'n_estimators': [100*i for i in range(1, 2)],
            'max_depth': [2*i for i in range(1, 2)],
            # 'scale_pos_weight': [2], # about 35% samples of increased mental metric 
            # 'scale_pos_weight' will not be used when n_job is not 0. Weird. 
            'learning_rate': [0.02*i for i in range(1, 2)],
            'objective': ['binary:hinge'],
            'eval_metric': ['error'],
            'tree_method': ['gpu_hist'],
            'gpu_id': [2],
            'subsample': [0.5+0.1*i for i in range(0, 1+1)],
            # 'colsample_bytree': [0.1*i for i in ranege(1, 10)],
            # 'min_child_weight': [2*i for i in range(1, 10)],
            # 'gamma': [0.1*i for i in range(1, 10)],
        }
    },
    "svm_rbf":{
            'estimator': sklearn.svm.SVC(kernel='rbf'),
#               Test grid
            'param_grid':   {
                'C':       [10**i for i in range(-1, 1)], 
                'class_weight': ['balanced'],
                'gamma':   [10**i for i in range(-1, 1)]
            }
    },
#     "rf":{
#         'estimator': sklearn.ensemble.RandomForestClassifier(),
# #            Test grid
#         'param_grid':   {
#             'n_estimators': [10*i for i in range(1, 2)],
#             'max_depth':     [2*i for i in range(1, 1+1)],
#         }
#     }, 
    "knn":{
        'estimator': sklearn.neighbors.KNeighborsClassifier(),
#             Full grid 
        'param_grid':   {
            'n_neighbors': [i for i in range(1, 30+1, 10)], 
            'weights': ['distance'], 
            'algorithm': ['ball_tree'],    
        }
    },

    
}

full_cfgs = {
#     "nn":{
#         'estimator': sklearn.neural_network.MLPClassifier(shuffle=True),
# #             Full grid
#         'param_grid':   {
#             'hidden_layer_sizes': gen_NN_uni(4, 100, 1, 20),  
#             'activation':         ['tanh',  'relu'], 
#             'max_iter':           [5000], 
#             'learning_rate':      ['adaptive']
#         }                
#     },
    "xgboost" : {
        'estimator': xgboost.XGBClassifier(),
        'param_grid': {
            'n_estimators': [100*i for i in range(1, 10)],
            'max_depth': [2*i for i in range(1, 10)],
            # 'scale_pos_weight': [2], # about 35% samples of increased mental metric 
            # 'scale_pos_weight' will not be used when n_job is not 0. Weird. 
            'learning_rate': [0.02*i for i in range(1, 15)],
            'objective': ['binary:hinge'],
            'eval_metric': ['error'],
            'tree_method': ['gpu_hist'],
            'gpu_id': [2],
            'subsample': [0.5+0.1*i for i in range(0, 5+1)],
            # 'colsample_bytree': [0.1*i for i in ranege(1, 10)],
            # 'min_child_weight': [2*i for i in range(1, 10)],
            # 'gamma': [0.1*i for i in range(1, 10)],
        }
    },
    "svm_rbf":{
        'estimator': sklearn.svm.SVC(kernel='rbf'),
#       Full grid
        'param_grid':   {
            'C':       [10**i for i in range(-8, 8)], 
            'class_weight': ['balanced'],
            'gamma':   [10**i for i in range(-8, 8)] # gamma gave me an error
        }
    },
#     "rf":{
#         'estimator': sklearn.ensemble.RandomForestClassifier(),
# #             Full grid 
#         'param_grid':   {
#             'n_estimators': [10*i for i in range(1, 20)],
#             'max_depth':     [2*i for i in range(20)], 
#             'max_samples': [0.05*i for i in range(1, 10+1)] # max samples gave me an error
#         }
#     }, 
#     "knn":{
#         'estimator': sklearn.neighbors.KNeighborsClassifier(),
# #             Full grid 
#         'param_grid':   {
#             'n_neighbors': [i for i in range(1, 30+1)], 
#             'weights': ['distance'], 
#             'algorithm': ['ball_tree', 'kd_tree'], 
#             'leaf_size': [5*i for i in range(1, 10+1)]                
#         }
#     }
}

if __name__ == "__main__":
    import pickle 

    import load_and_split # this project 

    # load and split 
    [X, Y] = pickle.load(open('../XY.pickle', 'br'))

    for target_index in range(3):
        print ("target index:", target_index,)
        train_X, train_y = load_and_split.prepare_training_data(X, Y, target_index=2)

        # CV 
        # import sk_cv # this project 
        search_cfg = full_cfgs # test_cfgs or full_cfgs
        for estimator_type, cfg in search_cfg.items():
            search_result = cv_trial(train_X, train_y, cfg)
            print ("\n", estimator_type, search_result)