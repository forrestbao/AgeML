import xgboost 
import pickle 

import load_and_split # this project 

from collections import Counter

import sys 

sys.path.append("../") # add parent directory to path
import load_and_split # this project 
import preprocessing # this project

result_dict = {} 
result_dict_pickle = 'xgboost_tree.pickle'

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

for use_feature_categories in feature_combinations:

    # load and split 
    [X,Y] = preprocessing.main(use_feature_categories, csv_file="../ML_social.csv", dump=True) 
    
    for target_index in range(4):
        print ("target index:", target_index,)
        train_X, train_y = load_and_split.prepare_training_data(X, Y, 
                                target_index=target_index,
                                minimal_length=4)

        # CV 
        progress = dict()
        watchlist  = [(train_X,'train-score')]
        bst = xgboost.XGBClassifier(n_estimators=300,
                                    max_depth=8, 
                                    scale_pos_weight=Counter(train_y)[0]/Counter(train_y)[1],
                                    learning_rate=0.1, 
                                    objective='binary:hinge',
                                    eval_metric='error',
                                    n_jobs=1,
                                    tree_method='gpu_hist', gpu_id=2)
        # fit model
        bst.fit(train_X, train_y) # , eval=watchlist, eval_result=progress)

        result_dict[(tuple(use_feature_categories), target_index)] = bst

pickle.dump(result_dict, open(result_dict_pickle, "wb"))

