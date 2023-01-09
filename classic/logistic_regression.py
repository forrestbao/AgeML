import pickle 
import sys 

sys.path.append("../") # add parent directory to path
import load_and_split # this project 
import preprocessing # this project

import sklearn.linear_model
import sklearn.model_selection

def lg_cv(train_X, train_y):
    model = sklearn.linear_model.LogisticRegressionCV(
        cv= sklearn.model_selection.ShuffleSplit(
            n_splits=5, test_size=0.2, random_state=0), 
        Cs=10, 
        solver='newton-cholseky',
        class_weight='balanced',
        n_jobs=10,
        
    )
    model.fit(train_X, train_y)
    return model


if __name__ == "__main__":

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

    for use_feature_categories in feature_combinations:

        # load and split 
        [X,Y] = preprocessing.main(use_feature_categories, csv_file="../ML_social.csv", dump=True)
        
        for target_index in range(4):
            print ("target index:", target_index,)
            train_X, train_y = load_and_split.prepare_training_data(X, Y, 
                                    target_index=target_index,
                                    minimal_length=4)


