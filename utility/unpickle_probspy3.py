# Method for unpickling probabilities/logits saved in process of evaluation.

import pickle

# Open file with pickled variables
def unpickle_probs(file, verbose = 0):
    with open(file, "rb") as f:  
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f, encoding='bytes')  # unpickle the content
        # the last part makes sure python 3 can open files pickled in python 2
        
    if verbose:    
        print("y_probs_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_probs_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels
        
    return ((y_probs_val, y_val), (y_probs_test, y_test))

# Convert python 3 pickled file to python 2 pickle
def convert_pickle32(file):
    with open(file, "rb") as f:
        w = pickle.load(f, encoding='bytes')

    pickle.dump(w, open(file,"wb"), protocol=2)
    
if __name__ == '__main__':
    
    (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(FILE_PATH, True)
    
    print(y_probs_val[:10])