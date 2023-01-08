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
            cv = sklearn.model_selection.ShuffleSplit(n_splits=5, test_size=0.2, random_state=0), 
            n_jobs=14,
            verbose=0, 
            return_train_score=True
        )

    # run grid search
    search.fit(training_X, training_y)
    print (search.best_params_, search.best_score_, search.cv_results_["mean_train_score"], search.cv_results_["std_train_score"])

    return search


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
    # "xgboost" : {
    #     'estimator': xgboost.XGBClassifier(),
    #     'param_grid': {
    #         'n_estimators': [100*i for i in range(2, 4)],
    #         'max_depth': [2*i for i in range(6, 6+1)],
    #         # 'scale_pos_weight': [2], # about 35% samples of increased mental metric 
    #         # 'scale_pos_weight' will not be used when n_job is not 0. Weird. 
    #         'learning_rate': [0.01*i for i in range(1, 5)],
    #         'objective': ['binary:hinge'],
    #         'eval_metric': ['error'],
    #         'tree_method': ['gpu_hist'],
    #         'gpu_id': [2],  # 1-indexed, 1 is the first GPU
    #         'subsample': [0.5+0.1*i for i in range(1, 1+1)],
    #         # 'colsample_bytree': [0.1*i for i in ranege(1, 10)],
    #         # 'min_child_weight': [2*i for i in range(1, 10)],
    #         # 'gamma': [0.1*i for i in range(1, 10)],
    #     }
    # },
#     "svm_rbf":{
#             'estimator': sklearn.svm.SVC(kernel='rbf'),
# #               Test grid
#             'param_grid':   {
#                 'C':       [10**i for i in range(5, 5+1)], 
#                 'class_weight': ['balanced'],
#                 'gamma':   [10**i for i in range(-5, -4)]
#             }
#     },
    "rf":{
        'estimator': sklearn.ensemble.RandomForestClassifier(),
#            Test grid
        'param_grid':   {
            'n_estimators': [10*i for i in range(1, 2)],
            'max_depth':     [2*i for i in range(1, 1+1)],
        }
    }, 
#     "knn":{
#         'estimator': sklearn.neighbors.KNeighborsClassifier(),
# #             Full grid 
#         'param_grid':   {
#             'n_neighbors': [i for i in range(1, 30+1, 10)], 
#             'weights': ['distance'], 
#             'algorithm': ['ball_tree'],    
#         }
#     },

    
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
    # "xgboost" : {
    #     'estimator': xgboost.XGBClassifier(),
    #     'param_grid': {
    #         'n_estimators': [100*i for i in range(4, 9)],
    #         'max_depth': [10*i for i in range(10, 100)],
    #         # 'scale_pos_weight': [2], # about 35% samples of increased mental metric 
    #         # 'scale_pos_weight' will not be used when n_job is not 0. Weird. 
    #         'learning_rate': [0.01*i for i in range(1, 15)],
    #         'objective': ['binary:hinge'],
    #         'eval_metric': ['error'],
    #         'tree_method': ['gpu_hist'],
    #         'gpu_id': [2],
    #         'subsample': [0.5+0.1*i for i in range(0, 5+1)],
    #         # 'colsample_bytree': [0.1*i for i in ranege(1, 10)],
    #         # 'min_child_weight': [2*i for i in range(1, 10)],
    #         # 'gamma': [0.1*i for i in range(1, 10)],
    #     }
    # },
    "svm_rbf":{
            'estimator': sklearn.svm.SVC(kernel='rbf'),
#               Test grid
            'param_grid':   {
                'C':       [10**i for i in range(1, 10)], 
                'class_weight': ['balanced'],
                'gamma':   [10**i for i in range(-10, 0)]
            }
    },
    "rf":{
        'estimator': sklearn.ensemble.RandomForestClassifier(),
#             Full grid 
        'param_grid':   {
            'n_estimators': [100*i for i in range(1, 10+1)],
            'max_depth':     [2*i for i in range(4, 20)], 
            # 'max_samples': [0.05*i for i in range(1, 10+1)] # max samples gave me an error
        }
    }, 
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
    import sys 

    sys.path.append("../") # add parent directory to path
    import load_and_split # this project 
    import preprocessing # this project

    CV_result_dict = {} 
    CV_result_dict_pickle = "CV_result_dict.pickle"

    # config 
    feature_combinations = [
        ["socio-demographics"],
        ["health"],
        ["social"],
        ["neighbourhood"],
        # ["socio-demographics", "health", "social", "neighbourhood"]
        ["health", "social"],
        ["social", "neighbourhood"],
        ["socio-demographics", "health"]
    ]
    # search_cfg = test_cfgs # test_cfgs or full_cfgs
    search_cfg = full_cfgs # test_cfgs or full_cfgs

    for use_feature_categories in feature_combinations:

        # load and split 
        [X,Y] = preprocessing.main(use_feature_categories, csv_file="../ML_social.csv", dump=True)
        
        for target_index in range(4):
            print ("target index:", target_index,)
            train_X, train_y = load_and_split.prepare_training_data(X, Y, 
                                    target_index=target_index,
                                    minimal_length=4)

            # CV 
            # import sk_cv # this project 

            for estimator_type, cfg in search_cfg.items():
                print ("\n", estimator_type)
                search = cv_trial(train_X, train_y, cfg)
                CV_result_dict[(tuple(use_feature_categories), target_index, estimator_type)] = search 
            # `search` is a GridSearchCV object.  

            # search.best_params_, search.best_score_, search.cv_results_.mean_train_score, search.cv_results_.std_train_score

    pickle.dump(CV_result_dict, open(CV_result_dict_pickle, "wb"))