import time
import pickle
import argparse
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

feat_to_use = []           # Indices of the features to use. If n is the number of features, from 0 to n-1. Apply both to train and test sets
class_index = -1           # Index of the class label. Apply both to train and test sets
debug = True

def load_features_and_class(filepath):
    ''' Load the features and the class indices from a .txt file 
        First line: features, second line: class
       
        Attributes:
            filepath (string)   :  Path to the .txt file
    '''
    with open(filepath, 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            tokens = line.strip().split(' ')
            if line_index == 0:
                global feat_to_use
                feat_to_use = [int(t) for t in tokens]
            elif line_index == 1:
                global class_index
                class_index= int(tokens[0])

def read_data(filepath):
    ''' Load a labelled point cloud from a .txt file

        Attributes:
            filepath (string)   :  Path to the .txt
        
        Return:
            X (np.array)   : Point cloud and features
            Y (np.array)   : Classes
    '''
    X, Y = [], []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split(' ')
            if 'nan' not in tokens:   
                X.append([float(t) for t_index, t in enumerate(tokens) if t_index != class_index])
                Y.append(int(float(tokens[class_index])))
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

def train_model(X_train, Y_train, n_estimators, max_depth, n_jobs):
    ''' Train the Random Forest model with the specified parameters and return it

        Attributes:
            X_train (np.array)  :   Training data
            Y_train (np.array)  :   Training classes
            n_estimators (int)  :   Number of trees in the forest
            max_depth (int)     :   Maximum depth of each tree
            n_jobs (int)        :   Number of threads used to train the model
        
        Return:
            model (np.RandomForestClassifier)  :   trained model
    '''
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0, oob_score=True, n_jobs=n_jobs)
    model.fit(X_train[:, feat_to_use], Y_train)         # Use only the specified features. 
    return model

def write_classification(X_test, Y_test_pred, filename):
    ''' Write a classified point cloud as a .txt file

        Attributes:
            X (np.array)        :   Point cloud and features
            Y (np.array)        :   Classes
            filename (string)   :   Output file path
    '''
    with open('{}.txt'.format(filename), 'w') as out:
        X = X.tolist()
        Y_pred = Y.tolist()
        for index, x in enumerate(X):
            x_as_str = " ".join([str(i) for i in x])
            out.write('{} {}\n'.format(x_as_str, str(Y_pred[index])))

def save_model(model, filename):
    ''' Save the trained machine learning model as .pkl file

        Attribures:
            model (np.RandomForestClassifier)   :   Model to save
            filename (string)                   :   Model output file
    '''
    with open(filename, 'wb') as out:
        pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(description='Train the random forest model.')
    parser.add_argument('features_filepath', help='Path to the file containing the index of the features and the class index')
    parser.add_argument('training_filepath', help='Path to the training file (.txt) [f1, ..., fn, c]')
    parser.add_argument('test_filepath', help='Path to the test file (.txt) [f1, ..., fn, c]')
    parser.add_argument('n_jobs', help='Number of threads used to train the model', type=int)
    parser.add_argument('output_name', help='Name of the predicted test file')
    args = parser.parse_args()
   
    print("Loading data...")
    load_features_and_class(args.features_filepath)
    X_train, Y_train = read_data (args.training_filepath)
    X_test, Y_test = read_data(args.test_filepath)
    print('\tTraining samples: {}\n\tTesting samples: {}\n\tUsing features with indices: {}'.format(len(Y_train), len(Y_test), feat_to_use))

    ''' ***************************************** TRAINING ************************************** '''
    # Parameters to test
    n_estimators = [50, 100, 150, 200]
    max_depths = [None]
    
    # Best configuration
    best_conf = {'ne' : 0, 'md' : 0} 
    best_f1 = 0

    print('\nTraining the model...')  
    start = time.time()                                  
    for ne, md in list(itertools.product(n_estimators, max_depths)):    # Train the model with different parameters and pick the one having the maximum f1-score on the test-set
        model = train_model(X_train, Y_train, ne, md, args.n_jobs)      # Train the model
        
        Y_test_pred = model.predict(X_test[:, feat_to_use])             # Test the model, using only the specified features
        
        acc = accuracy_score(Y_test, Y_test_pred)                       # Compute metrics and update best model
        f1 = f1_score(Y_test, Y_test_pred, average='weighted')
        if f1 > best_f1:                                                # Update best configuration
            best_conf['ne'] = ne
            best_conf['md'] = md
            best_f1 = f1
        
        if debug: print('\tne: {}, md: {} - acc: {} f1: {} oob_score: {}'.format(ne, md, acc, f1, model.oob_score_))
    end = time.time()
    print('---> Best parameters: ne: {}, md: {}'.format(best_conf['ne'], best_conf['md']))
    print('---> Feature importance:\n{}'.format(model.feature_importances_))
    print('---> Confusion matrix:\n{}'.format(confusion_matrix(Y_test, Y_test_pred)))
    print('---> Training time: {} seconds'.format(end - start))
    ''' ******************************************************************************************** '''
    
    # Save best model and write the best classification of the test set
    model = train_model(X_train, Y_train, best_conf['ne'], best_conf['md'], args.n_jobs)
    Y_test_pred = model.predict(X_test[:, feat_to_use])
    write_classification(X_test, Y_test_pred, args.output_name)
    save_model(model, 'ne{}_md{}.pkl'.format(best_conf['ne'], best_conf['md']))


if __name__== '__main__':
    main()