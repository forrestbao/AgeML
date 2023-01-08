# %% [markdown]
# # Training data preparation
# In this project, we have human subject data sampled annually. The human subjects are identified uniquelly via the variable `hhidpn` and the year via `wave`. 
# * Features: 
#     * categorical: `race`, 
#     * semi-categorical/numerical: `educ`, `cohort`, `female` (binary)
#     * continous/numerical: `age`, `activity`, `freqpray`, `mwi`, `assets`, `income`, `social` (?), `physd` (?), `formal`, `informal`, `depress`, `memoryp` (binary), `everyday`, `majordisc`, `cumulative`,  
#     * not in use: `wave` (year), `dead` (binary), 
# * Targets: `recall`, `seven`, `bwcount`. they are added into `cogtot`. for most people, they all drop along time as our brain degrades. 
# 
# Some variables are not time-variant: `female`, `race`, `educ`, `cohort`. 
# 

import pickle 
import typing 

import pandas
import numpy
import tqdm
import sklearn, sklearn.preprocessing 

# %%
def remove_short_wave_subjects(df):
    print ("Number of subjects before wave removal:", len(df['hhidpn'].unique()), "Number of rows:", len(df) )

    okay_row, okay_subject, seq_length = 0, 0,  []  

    qualified_subject = []
    for subjectID in tqdm.tqdm(df['hhidpn'].unique()):
        subject_df = df[df['hhidpn']==subjectID]
        if len(subject_df) > 1:
            okay_subject += 1
            okay_row += len(subject_df) 
            seq_length.append(len(subject_df) )
            qualified_subject.append(int(subjectID)) 

            # debug 
            # if okay_subject> 10 :
            #     break 

    df = df[df['hhidpn'].isin(qualified_subject)] 

    print ("\nNumber of >2-wave subjets:", okay_subject, "Number of >2-wave rows:", okay_row, "average length (years) of a subject:",  sum(seq_length)/len(seq_length), "average age of subjects", df['age'].mean() )

    return df 

def clean_columns_and_rows(df, keep_columns, target_columns):
    # drop unecessary columns 
    df = df[keep_columns]
    print ("Columns kept:", df.columns)

    print (f"Number of rows before dropping: {len(df)}, and subjects: {len(df['hhidpn'].unique())}")
    # Drop rows where dead==1 or age<=0
    df = df[df['dead']==0]
    print (f"After dropping rows where dead column is 0,  \n\t in remaining data, \
            number of rows:  {len(df)}, and subjects: {len(df['hhidpn'].unique())} ")
    df = df[df['age']>0]
    print (f"After dropping rows where age column is not a number: \n\t in remaining data, \
            number of rows:  {len(df)}, and subjects: {len(df['hhidpn'].unique())} ")

    # Replace special strings with NaN 
    for special_string in [".s", ".m", ".r", ".d", ".x"]:
        df = df.replace(special_string, numpy.nan)

    for column in df.columns: 
        # print (type(df_test[column].dtype))
        if df[column].dtype == numpy.object_:
            # print (column)
            df[column] = pandas.to_numeric(df[column])

    # Remove rows that has NaN in target columns 
    print (f"Original number of rows: {len(df)}, and subjects: {len(df['hhidpn'].unique())}") 
    df=df.dropna(subset=target_columns)
    print (f"After dropping rows that has NaN in target columns, \
            \n\t in remaining data, \
            number of rows:  {len(df)}, and subjects: {len(df['hhidpn'].unique())} ")

    # Remove subjects with less than 2 waves
    df = remove_short_wave_subjects(df)

    # fill N/A
    df = df.fillna(0.5)

    return df


# %%
# Convert the dataframe into arraies:
# [
#     [# subject 1 
#         [feature_1, feature_2, feature_3], # year 1 
#         [], # year 2
#         ...
#         [], # year M  
#     ], 
#     [
#         [], # year 1 
#         [], # year 2
#         ...
#         [], # year M  
#     ], # subject 2
#     ...
#     [
#         [], # year 1 
#         [], # year 2
#         ...
#         [], # year M  
#     ], # subject N  
# ]


def pack_into_time_series(df, numerical_features, categorical_features, target_columns):
    X, Y = [], [] # X and y are the training inputs and targets. 
    # X, Y = numpy.array([]), numpy.array([]) # X and y are the training inputs and targets. 
    # X or Y is a 1D list of 2D numpy arrays. X[subject] -> array[year][feature]

    feature_columns = numerical_features

    # one hot encoding for categorical features
    one_hot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
    if categorical_features != []:
        categorical_transformed = one_hot_encoder.fit_transform(df[categorical_features].to_numpy()) 
        one_hot_encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_features)
        ohe_df = pandas.DataFrame(categorical_transformed, 
                                    columns=one_hot_encoded_feature_names, 
                                    index = df.index) # without the index, concat misalin unless using ignore_index
        feature_columns = numerical_features + one_hot_encoded_feature_names.tolist()
        df = df.drop(categorical_features, axis=1)
        df = pandas.concat([df, ohe_df], axis=1)
      
    # # Columns of type Object to float64 
    # for a_numerical_feature in numerical_features: 
    #     if df[a_numerical_feature].dtype not in ['int', 'float']:
    #         df[a_numerical_feature] = df[a_numerical_feature].astype(float)

    # min-max scale for numerical features 
    feature_scaler = sklearn.preprocessing.MinMaxScaler()
    df[numerical_features] = feature_scaler.fit_transform(df[numerical_features])

    target_scaler = sklearn.preprocessing.MinMaxScaler()
    df[target_columns] = target_scaler.fit_transform(df[target_columns])

    for subjectID in tqdm.tqdm(df['hhidpn'].unique()):
        subject_df = df[df['hhidpn'] == subjectID]
        subject_X, subject_Y = subject_df[feature_columns].to_numpy(), subject_df[target_columns].to_numpy()
        X.append(subject_X)
        Y.append(subject_Y)
        # numpy.append(X, subject_X, axis=0)
        # numpy.append(Y, subject_Y, axis=0)
        # break 

    # X, Y  = numpy.array(X), numpy.array(Y)

    return X, Y

def main(use_feature_categories, csv_file="ML_social.csv", dump=False):

    feature_categories  = {
        "socio-demographics": ["race", "educ", "cohort", "female", "age"], # "income", "assets"],
        "health": ["mwi", "depress", "memoryp"], # Ernest and Cliff suggested to exclude memoryp
        "social": ["activity", "freqpray", "formal", "informal"], # "everyday", "majordisc", "cumulative"
        "neighbourhood": ["social", "physd"],
    }

    # to exclude: everyday, majordisc, cumulative 
    # [gender, age, education]
    # [mwi, depress]
    # [everyday]

    numerical_features = [] 
    for category in use_feature_categories:
        numerical_features += feature_categories[category] 

    categorical_features = []
    if "socio-demographics" in use_feature_categories:
        numerical_features.remove("race")
        categorical_features = ["race"]

    # feature_columns = numerical_features + categorical_features

    target_columns = ["cogtot", "recall", "seven", "bwcount"]
    identifier_columns = ["hhidpn", "wave", "dead"]
    keep_columns = numerical_features + categorical_features + target_columns +  identifier_columns

    if "age" not in keep_columns:
        keep_columns.append("age") # age is used to filter rows 

    df = pandas.read_csv(csv_file)
    df = clean_columns_and_rows(df, keep_columns, target_columns)
    X, Y  = pack_into_time_series(df, numerical_features, categorical_features, target_columns)
    print (X[0].shape, Y[0].shape)
    if dump:
        feature_names= "_".join(use_feature_categories)
        pickle.dump([X, Y], open(f"XY_{feature_names}.pickle", 'wb'))

    return [X, Y]

if __name__ == "__main__":

    use_feature_categories = ["socio-demographics", "health", "social", "neighbourhood"]
    use_feature_categories = ["socio-demographics", "health"]

    [X,Y] = main(use_feature_categories, dump=True) 

# %%

