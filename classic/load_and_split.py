# Load cleaned data and split into training pairs 

#%%
import numpy
import einops

# %%

def slide_over_one_subject(subject_X, subject_Y, minimal_length = 4):
    """Create samples from the data of one subject using a sliding window

    subject_X, subject_Y: 2D numpy array, rows for waves and columns for features
    subject_new_X, subject_new_Y: 3D numpy array, axis 0 for sliding window, 
                                  axis 1 for waves, axis 2 for features

    """
    subject_new_X, subject_new_Y = [], []
    num_features = subject_X.shape[1]
    subject_X_windowed = numpy.lib.stride_tricks.sliding_window_view(
        subject_X, 
        window_shape=(minimal_length, num_features)
    ) 
    subject_new_X = subject_X_windowed[:, -1, :, :]

    num_targets = subject_Y.shape[1]
    subject_Y_windowed = numpy.lib.stride_tricks.sliding_window_view(
        subject_Y, 
        window_shape=(minimal_length, num_targets)
    ) 
    subject_new_Y = subject_Y_windowed[:, -1, :, :]

    return subject_new_X, subject_new_Y

def flatten_X(X, by:str):
    """
    X, Y: 1-D list of 2D num array, each element is the data of one subject
        X[i].shape[0] === Y[i].shape[0] === number of waves

    Verification of the syntax: 
    X=numpy.array([[[1,2,3], [4,5,6]], 
              [[7,8,9], [10,11,12]], 
              [[13,14,15], [16,17,18]], 
              [[19,20,21], [22,23,24]]])

    einops.rearrange(X, 's w f -> s (f w)')

    array([[ 1,  4,  2,  5,  3,  6],
       [ 7, 10,  8, 11,  9, 12],
       [13, 16, 14, 17, 15, 18],
       [19, 22, 20, 23, 21, 24]])

    einops.rearrange(X, 's w f -> s (w f)')

    array([[ 1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, 10, 11, 12],
           [13, 14, 15, 16, 17, 18], 
           [19, 20, 21, 22, 23, 24]])

    """
    # [num_subject, num_wave, num_feature] = X.shape
    # return X.reshape(-1, num_feature)

    if by == 'wave':
        new_X = einops.rearrange(X, 's w f -> s (f w)') # s: subject, w: wave, f: feature
        # Concatenate based on wave
    elif by == 'feature':
        new_X = einops.rearrange(X, 's w f -> s (w f)') # s: subject, w: wave, f: feature
        # Concatenate based on feature 
    return new_X

def sampling_fixed_length(X, Y, minimal_length = 4):
    """

    X, Y: 1-D list of 2D num array, each element is the data of one subject
          X[i].shape[0] === Y[i].shape[0] === number of waves

    """
    new_X, new_Y = [], []
    for subject_X, subject_Y in zip(X, Y):
        if len(subject_Y)>= minimal_length:
            subject_new_X, subject_new_Y = slide_over_one_subject(subject_X, subject_Y, minimal_length=minimal_length)
            new_X += subject_new_X.tolist()
            new_Y += subject_new_Y.tolist()

    new_X, new_Y = numpy.array(new_X), numpy.array(new_Y)

    return new_X, new_Y

def generate_labels(Y, target_index: int):
    """Extract the training label for a particular target dimension 

    Y: 3D ndarray, axis 0 is sample, axis 1 is wave, axis 2 is score 
    """
    y_of_interest = Y[:, :, target_index] # 2D array 
    begin_score = Y[:, 0, target_index]
    end_score = Y[:, -1, target_index]

    labels = end_score - begin_score
    labels = numpy.heaviside(labels, 0)

    return labels.reshape((-1,1)) # 2d arrary, Nx1

# top level function for training data preparation 
# need to rerun for each target index rangeing from 0 to 2 
def prepare_training_data(X, Y, target_index: int, minimal_length=4):
    """

    X, Y: 1-D list of 2D num array, each element is the data of one subject
          X[i].shape[0] === Y[i].shape[0] === number of waves

    """
    new_X, new_Y = sampling_fixed_length(X, Y, minimal_length=minimal_length) 
    new_Y = generate_labels(new_Y, target_index=target_index)
    new_X = flatten_X(new_X, by='feature')

    # to cope with Scikit-learn's preference to Y 
    new_Y = new_Y.reshape((-1,))
    return new_X, new_Y 

# %%

if __name__ == "__main__":
    import pickle 
    [X, Y] = pickle.load(open('XY.pickle', 'br'))

    # there are three predictive columns, identified by target_index
    train_X, train_y = prepare_training_data(X, Y, target_index=2)
# %%
